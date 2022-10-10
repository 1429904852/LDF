import keras
from keras import Model
from keras.layers import Conv1D, Dense, Layer, Input
import tensorflow as tf
import random


class SupervisedCL(Layer):
    def __init__(self, N, K, conv_dim, temperature=0.1, alpha=0.1):
        super(SupervisedCL, self).__init__()
        self.N = N
        self.K = K
        self.conv_dim = conv_dim
        self.t = temperature
        self.alpha = alpha
        self.B = 1
        if N == 5 and K == 5:
            self.B = 2

    def call(self, inputs, **kwargs):
        op = inputs
        proto_rep_all_classes = tf.squeeze(op, axis=-2)  # B, N, K, conv_dim
        proto_norm = tf.math.reduce_euclidean_norm(proto_rep_all_classes, axis=-1, keepdims=True)
        proto_rep_all_classes = proto_rep_all_classes / proto_norm
        proto_rep_list = tf.split(value=proto_rep_all_classes, num_or_size_splits=self.N, axis=1)
        proto_rep_all_classes = tf.concat(proto_rep_list, axis=2)  # B, 1, N*K, conv_dim
        proto_rep_all_classes = tf.squeeze(proto_rep_all_classes, axis=1)  # B, N*K, conv_dim
        cl_loss = 0.0
        for i in range(self.N):
            proto_rep_one_class = tf.split(value=tf.squeeze(proto_rep_list[i], axis=1), num_or_size_splits=self.K,
                                           axis=1)
            for j in range(self.K):
                anchor = proto_rep_one_class[j]  # B, 1, conv_dim
                exp_first_term = tf.math.exp(
                    tf.matmul(anchor, proto_rep_all_classes, transpose_b=True) / self.t)  # B, 1, N*K
                exp_second_term = tf.math.exp(
                    tf.squeeze(tf.matmul(anchor, anchor, transpose_b=True) / self.t, axis=1))  # B, 1
                exp_term = tf.reduce_sum(exp_first_term, axis=-1) - exp_second_term
                log_term = tf.math.log(exp_term)  # B, 1
                sum_first_term = tf.matmul(anchor, tf.concat(proto_rep_one_class, axis=1),
                                           transpose_b=True) / self.t  # B, 1, K
                sum_second_term = tf.squeeze(tf.matmul(anchor, anchor, transpose_b=True) / self.t, axis=1)  # B, 1
                sum_term = (tf.reduce_sum(sum_first_term, axis=-1) - sum_second_term) / (self.K - 1)  # B, 1
                loss_for_one_anchor = log_term - sum_term  # B, 1
                loss_for_one_anchor = tf.squeeze(loss_for_one_anchor, axis=-1)  # shape = (B, )
                loss_for_one_anchor = tf.reduce_sum(loss_for_one_anchor, axis=-1)  # a scalar
                # compute total cl loss:
                cl_loss += loss_for_one_anchor
        self.add_loss(self.alpha * cl_loss / (self.B * self.N * self.K * self.conv_dim))
        return inputs


def make_model(N, K, Q, Em, max_len, alpha=0.1, temp=0.1, word_embedding_dim=50, conv_dim=50, kernel_size=3):

    SupervisedContrastiveLoss = SupervisedCL(N=N, K=K, conv_dim=conv_dim, temperature=temp, alpha=alpha)
    s_input = Input(shape=(N, K, max_len, word_embedding_dim))
    q_input = Input(shape=(N, Q, max_len, word_embedding_dim))
    s_mask = Input(shape=(N, K, max_len))
    q_mask = Input(shape=(N, Q, max_len))
    s_sentence = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                        bias_initializer=keras.initializers.RandomNormal(stddev=0.1))(s_input)

    s_sentence = s_sentence * tf.tile(tf.expand_dims(s_mask, axis=-1), multiples=[1, 1, 1, 1, conv_dim])

    q_sentence = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                        bias_initializer=keras.initializers.RandomNormal(stddev=0.1))(q_input)

    # compute Ws
    Vs = tf.reduce_mean(tf.reduce_mean(s_sentence, axis=2), axis=2) / tf.reduce_mean(tf.reduce_mean(s_mask, axis=-1),
                                                                                     axis=-1,
                                                                                     keepdims=True)  # B, N, conv dim
    Vs_Em_times = tf.tile(tf.expand_dims(Vs, axis=2), multiples=[1, 1, Em, 1])  # B, N, Em, conv dim
    Vs_Em_times = tf.transpose(Vs_Em_times, perm=[0, 1, 3, 2])  # B, N, conv dim, Em
    Ws = Dense(units=conv_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))(Vs_Em_times)
    # B, N, conv dim, conv dim
    # compute prototype
    beta_raw = tf.matmul(s_sentence, tf.expand_dims(Ws, axis=2))
    beta_raw = tf.tanh(beta_raw)  # B, N, K, max len, conv dim
    beta_final = tf.matmul(beta_raw, tf.expand_dims(tf.expand_dims(Vs, axis=2), axis=-1))  # B, N, K, max len, 1
    beta_final = tf.squeeze(beta_final, axis=-1, name='before_where')  # B, N, K, max len s_mask[0][0][0][:5] == 0
    beta_final = tf.where(condition=tf.cast(s_mask, dtype=bool), x=beta_final,
                          y=tf.constant(value=-1e9, shape=(1, N, K, max_len)), name='where')
    beta_final = tf.nn.softmax(beta_final, axis=-1)  # B, N, K, max len

    proto_raw = tf.matmul(tf.expand_dims(beta_final, axis=-2), s_sentence)  # B, N, K, 1, conv dim
    proto_raw = SupervisedContrastiveLoss(proto_raw)
    proto_final = tf.reduce_mean(tf.squeeze(proto_raw, axis=-2), axis=2)  # B, N, conv dim
    # compute query attention
    q_sentence = tf.tanh(q_sentence)  # B, N, Q, max len, conv dim

    pho_raw = tf.matmul(q_sentence,
                        tf.transpose(tf.expand_dims(tf.expand_dims(proto_final, axis=1), axis=1), perm=[0, 1, 2, 4, 3]))
    q_mask_tiled = tf.tile(tf.expand_dims(q_mask, axis=-1), multiples=[1, 1, 1, 1, N])
    pho_raw = tf.where(condition=tf.cast(q_mask_tiled, dtype=bool), x=pho_raw,
                       y=tf.constant(value=-1e9, shape=(1, N, Q, max_len, N)))
    pho = tf.nn.softmax(logits=pho_raw, axis=-2)
    # pho: B, N, Q, max len, N
    query_att = tf.matmul(tf.transpose(pho, perm=[0, 1, 2, 4, 3]), q_sentence)  # B, N, Q, N, conv dim

    # compute distances
    ED = tf.math.reduce_euclidean_norm((query_att - tf.expand_dims(tf.expand_dims(proto_final, axis=1), axis=1)),
                                       axis=-1)
    y = tf.nn.softmax(logits=-ED, axis=-1)  # B, N, Q, N

    model_awatt = Model(inputs=[s_input, q_input, s_mask, q_mask], outputs=y)

    return model_awatt



