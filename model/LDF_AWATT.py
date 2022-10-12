import keras
from keras import Model
from keras.layers import Conv1D, Dense, Layer, Input
import tensorflow as tf
import random


class LabelCL(Layer):  # B, N, K, 1, conv dim---prototype raws
    def __init__(self, N, K, conv_dim, temperature=0.1, alpha=0.1):
        super(LabelCL, self).__init__()
        self.N = N
        self.K = K
        self.conv_dim = conv_dim
        self.t = temperature
        self.alpha = alpha
        self.B = 1
        if N == 5 and K == 5:
            self.B = 2

    def call(self, inputs, **kwargs):
        proto_raw, class_input = inputs
        op_1 = proto_raw
        op_2 = class_input

        proto_rep_all_classes = tf.squeeze(op_1, axis=-2)  # B, N, K, conv_dim
        proto_norm = tf.math.reduce_euclidean_norm(proto_rep_all_classes, axis=-1, keepdims=True)
        proto_rep_all_classes = proto_rep_all_classes / proto_norm

        class_name = tf.reduce_mean(op_2, axis=-2)  # B, N, word_embedding_dim
        class_norm = tf.math.reduce_euclidean_norm(class_name, axis=-1, keepdims=True)
        class_name = class_name / class_norm  # B, N, word_embedding_dim
        class_cos_weight = tf.matmul(class_name, class_name, transpose_b=True)  # B, N, N
        # first N stands for N classes, and the second N stands for the N cos similarities between classes for one class
        class_cos_weight = tf.tile(tf.expand_dims(class_cos_weight, axis=-1), multiples=[1, 1, 1, self.K])
        cos_weight_list = tf.split(value=class_cos_weight, num_or_size_splits=self.N, axis=1)
        # (B, 1, N, K) x N

        proto_rep_list = tf.split(value=proto_rep_all_classes, num_or_size_splits=self.N, axis=1)
        # tf.split returns a list, every element now is of shape: B, 1, K, conv_dim, element num: N

        proto_rep_all_classes = tf.reshape(proto_rep_all_classes, shape=(-1, self.N * self.K, self.conv_dim))
        # B, N*K, conv_dim
        cl_loss = 0.0
        for i in range(self.N):
            proto_rep_one_class = tf.split(value=tf.squeeze(proto_rep_list[i], axis=1), num_or_size_splits=self.K,
                                           axis=1)
            cos_one_class = tf.reshape(cos_weight_list[i], (-1, 1, self.N * self.K))
            # B, 1, NK
            # every element now for proto_rep_one_class is of shape: B, 1, conv_dim, element num: K
            for j in range(self.K):
                # which is : log (sum({a over A(i)}: exp(z_i·z_a/t))) - 1/|P(i)|·sum({p over P(i)}: z_i·z_p/t)
                anchor = proto_rep_one_class[j]  # B, 1, conv_dim
                # sum({a over A(i)}: exp(z_i·z_a/t)) = sum({a over all<A(i) + i>}: exp(z_i·z_a/t)) - exp(z_i·z_i/t)
                exp_first_term = tf.math.exp(
                    tf.matmul(anchor, proto_rep_all_classes, transpose_b=True) / self.t)  # B, 1, N*K

                exp_first_term = exp_first_term * cos_one_class  # B, 1, NK

                exp_second_term = tf.math.exp(
                    tf.squeeze(tf.matmul(anchor, anchor, transpose_b=True) / self.t, axis=1))  # B, 1
                exp_term = tf.reduce_sum(exp_first_term, axis=-1) - exp_second_term
                # B, 1
                # log (sum({a over A(i)}: exp(z_i·z_a/t)))
                log_term = tf.math.log(exp_term)  # B, 1

                # compute the second term:
                # sum({p over P(i)}: z_i·z_p/t) = sum({p over all positive<P(i) + i>}: exp(z_i·z_p/t)) - exp(z_i·z_i/t)
                sum_first_term = tf.matmul(anchor, tf.concat(proto_rep_one_class, axis=1),
                                           transpose_b=True) / self.t  # B, 1, K
                sum_second_term = tf.squeeze(tf.matmul(anchor, anchor, transpose_b=True) / self.t, axis=1)  # B, 1

                sum_term = (tf.reduce_sum(sum_first_term, axis=-1) - sum_second_term) / (self.K - 1)  # B, 1

                # compute cl loss for one anchor:
                loss_for_one_anchor = log_term - sum_term  # B, 1
                loss_for_one_anchor = tf.squeeze(loss_for_one_anchor, axis=-1)  # shape = (B, )
                loss_for_one_anchor = tf.reduce_sum(loss_for_one_anchor, axis=-1)  # a scalar
                # compute total cl loss:
                cl_loss += loss_for_one_anchor
        self.add_loss(self.alpha * cl_loss / (self.B * self.N * self.K * self.conv_dim))
        # we add the loss via add_loss method, and we don't change the input tensors (proto_raw, class_input)
        return proto_raw


def make_model(N, K, Q, Em, max_len, alpha=0.1, temp=0.1, word_embedding_dim=50, conv_dim=50, kernel_size=3):
    LabelContrastiveLoss = LabelCL(N=N, K=K, conv_dim=conv_dim, temperature=temp, alpha=alpha)
    s_input = Input(shape=(N, K, max_len, word_embedding_dim))
    q_input = Input(shape=(N, Q, max_len, word_embedding_dim))
    s_mask = Input(shape=(N, K, max_len))
    q_mask = Input(shape=(N, Q, max_len))
    class_input = Input(shape=(N, 10, word_embedding_dim))  # B, N, 10, word_embedding_dim

    s_sentence = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                        bias_initializer=keras.initializers.RandomNormal(stddev=0.1))(s_input)

    class_name = tf.reduce_mean(class_input, axis=-2)  # B, N, word_embedding_dim
    class_name = tf.expand_dims(tf.expand_dims(class_name, axis=-2), axis=-2)  # B, N, 1, 1, word_embedding_dim
    class_name = tf.tile(class_name, multiples=[1, 1, K, max_len, 1])  # B, N, K, max len, word_embedding_dim

    # compute cos similarity: x.Transpose · y/||x||*||y||
    s_input_processed = tf.where(condition=tf.cast(tf.tile(tf.expand_dims(s_mask, axis=-1),
                                                           multiples=[1, 1, 1, 1, word_embedding_dim]), dtype=bool),
                                 x=s_input, y=tf.constant(value=1.0, shape=(1, N, K, max_len, word_embedding_dim)))
    dot_product = tf.reduce_sum(class_name * s_input_processed, axis=-1)  # B, N, K, max len

    l2_norm = tf.math.reduce_euclidean_norm(class_name, axis=-1) * tf.math.reduce_euclidean_norm(s_input_processed,
                                                                                                 axis=-1)
    cos_similarity = dot_product / l2_norm

    s_sentence = s_sentence * tf.tile(tf.expand_dims(s_mask, axis=-1), multiples=[1, 1, 1, 1, conv_dim])

    q_sentence = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                        bias_initializer=keras.initializers.RandomNormal(stddev=0.1))(q_input)

    # compute Ws
    Vs = tf.reduce_mean(tf.reduce_mean(s_sentence, axis=2), axis=2) / tf.reduce_mean(tf.reduce_mean(s_mask, axis=-1),
                                                                                     axis=-1,
                                                                                     keepdims=True)  # B, N, conv dim
    # first reduce_mean: B, N, max len, conv dim,   second reduce_dim: B, N, conv dim
    Vs_Em_times = tf.tile(tf.expand_dims(Vs, axis=2), multiples=[1, 1, Em, 1])  # B, N, Em, conv dim
    Vs_Em_times = tf.transpose(Vs_Em_times, perm=[0, 1, 3, 2])  # B, N, conv dim, Em
    Ws = Dense(units=conv_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))(Vs_Em_times)
    # B, N, conv dim, conv dim

    # compute prototype
    beta_raw = tf.tanh(tf.matmul(s_sentence, tf.expand_dims(Ws, axis=2)))  # B, N, K, max len, conv dim
    beta_final = tf.matmul(beta_raw, tf.expand_dims(tf.expand_dims(Vs, axis=2), axis=-1))  # B, N, K, max len, 1
    # Vs after two expand dims: B, N, 1, conv dim, 1
    beta_final = tf.squeeze(beta_final, axis=-1, name='before_where')  # B, N, K, max len s_mask[0][0][0][:5] == 0

    beta_final = tf.stack([beta_final, cos_similarity], axis=-1)  # B, N, K, max len, 2
    beta_final = Dense(units=1)(beta_final)
    beta_final = tf.squeeze(beta_final, axis=-1)
    # beta_final = beta_final * cos_similarity  # B, N, K, max len
    beta_final = tf.where(condition=tf.cast(s_mask, dtype=bool), x=beta_final,
                          y=tf.constant(value=-1e9, shape=(1, N, K, max_len)), name='where')  # B, N, K, max len
    beta_final = tf.nn.softmax(beta_final, axis=-1)  # B, N, K, max len

    proto_raw = tf.matmul(tf.expand_dims(beta_final, axis=-2), s_sentence)  # B, N, K, 1, conv dim
    proto_raw = LabelContrastiveLoss([proto_raw, class_input])
    proto_final = tf.reduce_mean(tf.squeeze(proto_raw, axis=-2), axis=2)  # B, N, conv dim
    # compute query attention
    q_sentence = tf.tanh(q_sentence)  # B, N, Q, max len, conv dim

    pho_raw = tf.matmul(q_sentence,
                        tf.transpose(tf.expand_dims(tf.expand_dims(proto_final, axis=1), axis=1), perm=[0, 1, 2, 4, 3]))
    # pho_raw: B, N, Q, max len, N
    q_mask_tiled = tf.tile(tf.expand_dims(q_mask, axis=-1), multiples=[1, 1, 1, 1, N])
    pho_raw = tf.where(condition=tf.cast(q_mask_tiled, dtype=bool), x=pho_raw,
                       y=tf.constant(value=-1e9, shape=(1, N, Q, max_len, N)))
    pho = tf.nn.softmax(logits=pho_raw, axis=-2)
    # pho: B, N, Q, max len, N
    query_att = tf.matmul(tf.transpose(pho, perm=[0, 1, 2, 4, 3]), q_sentence)  # B, N, Q, N, conv dim

    # compute distances
    ED = tf.math.reduce_euclidean_norm((query_att - tf.expand_dims(tf.expand_dims(proto_final, axis=1), axis=1)),
                                       axis=-1)
    # B, N, Q, N
    y = tf.nn.softmax(logits=-ED, axis=-1)  # B, N, Q, N

    model_awatt = Model(inputs=[s_input, q_input, s_mask, q_mask, class_input], outputs=y)

    return model_awatt
