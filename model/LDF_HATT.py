import keras
import tensorflow as tf
from keras.layers import Dense, Input, Conv2D, Dropout, Layer
from keras import Model
from model import Encoder
import random


class LabelCL(Layer):  # B, N, K, conv dim---encoded support
    def __init__(self, N, K, conv_dim, temperature=0.1, alpha=0.1):
        super(LabelCL, self).__init__()
        self.N = N
        self.K = K
        self.conv_dim = conv_dim
        self.t = temperature
        self.alpha = alpha
        self.B = 1

    def call(self, inputs, **kwargs):
        proto_raw, class_input = inputs
        op_1 = proto_raw
        op_2 = class_input
        proto_rep_all_classes = tf.reshape(op_1, (-1, self.N, self.K, self.conv_dim))
        proto_norm = tf.math.reduce_euclidean_norm(proto_rep_all_classes, axis=-1, keepdims=True)
        proto_rep_all_classes = proto_rep_all_classes / proto_norm

        class_name = tf.reduce_mean(op_2, axis=-2)  # B, N, word_embedding_dim
        class_norm = tf.math.reduce_euclidean_norm(class_name, axis=-1, keepdims=True)
        class_name = class_name / class_norm  # B, N, word_embedding_dim
        class_cos_weight = tf.matmul(class_name, class_name, transpose_b=True)  # B, N, N
        # class_cos_weight = class_cos_weight * tf.expand_dims(tf.eye(self.N), axis=0)
        class_cos_weight = tf.tile(tf.expand_dims(class_cos_weight, axis=-1), multiples=[1, 1, 1, self.K])
        cos_weight_list = tf.split(value=class_cos_weight, num_or_size_splits=self.N, axis=1)
        # (B, 1, N, K) x N
        proto_rep_list = tf.split(value=proto_rep_all_classes, num_or_size_splits=self.N, axis=1)
        proto_rep_all_classes = tf.reshape(proto_rep_all_classes, shape=(-1, self.N * self.K, self.conv_dim))
        cl_loss = 0.0
        for i in range(self.N):
            proto_rep_one_class = tf.split(value=tf.squeeze(proto_rep_list[i], axis=1), num_or_size_splits=self.K,
                                           axis=1)
            cos_one_class = tf.reshape(cos_weight_list[i], (-1, 1, self.N * self.K))
            # B, 1, NK
            for j in range(self.K):
                anchor = proto_rep_one_class[j]  # B, 1, conv_dim
                exp_first_term = tf.math.exp(
                    tf.matmul(anchor, proto_rep_all_classes, transpose_b=True) / self.t)  # B, 1, N*K
                exp_first_term = exp_first_term * cos_one_class  # B, 1, NK

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
        return proto_raw


def make_model(N, K, Q, max_len, alpha=0.1, temp=0.1, word_embedding_dim=50, conv_dim=50, kernel_size=3):
    LabelContrastiveLoss = LabelCL(N=N, K=K, conv_dim=conv_dim, temperature=temp, alpha=alpha)
    shared_dense = Dense(units=conv_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
    shared_conv = Encoder.CNNEncoder(conv_dim=conv_dim, kernel_size=kernel_size)

    s_input = Input(shape=(N, K, max_len, word_embedding_dim))
    q_input = Input(shape=(N, Q, max_len, word_embedding_dim))
    s_mask = Input(shape=(N, K, max_len))
    q_mask = Input(shape=(N, Q, max_len))
    class_input = Input(shape=(N, 10, word_embedding_dim))  # B, N, 10, word_embedding_dim

    class_name = tf.reduce_mean(class_input, axis=-2)  # B, N, word_embedding_dim
    class_name = tf.expand_dims(tf.expand_dims(class_name, axis=-2), axis=-2)  # B, N, 1, 1, word_embedding_dim
    class_name = tf.tile(class_name, multiples=[1, 1, K, max_len, 1])  # B, N, K, max len, word_embedding_dim
    s_input_processed = tf.where(condition=tf.cast(tf.tile(tf.expand_dims(s_mask, axis=-1),
                                                           multiples=[1, 1, 1, 1, word_embedding_dim]), dtype=bool),
                                 x=s_input, y=tf.constant(value=1.0, shape=(1, N, K, max_len, word_embedding_dim)))
    dot_product = tf.reduce_sum(class_name * s_input, axis=-1)  # B, N, K, max len
    l2_norm = tf.math.reduce_euclidean_norm(class_name, axis=-1) * tf.math.reduce_euclidean_norm(s_input_processed,
                                                                                                 axis=-1)
    # print(l2_norm.shape)
    cos_similarity = dot_product / l2_norm
    cos_similarity = tf.reshape(cos_similarity, (-1, max_len, 1))
    # BNK, max len, 1
    cos_similarity = tf.tile(cos_similarity, multiples=[1, 1, conv_dim])
    # BNK, max len, conv dim
    # [B * N * K, seq_len, hidden_size]
    s_sentence = tf.reshape(s_input, (-1, max_len, word_embedding_dim))
    # BNK, max len, word embedding dim
    encoded_support = shared_conv(s_sentence)
    # [B * N * K, hidden_size]
    s_mask_processed = tf.reshape(s_mask, shape=(-1, max_len))
    s_mask_processed = tf.cast(tf.tile(tf.expand_dims(s_mask_processed, axis=-1), multiples=[1, 1, conv_dim]), bool)
    encoded_support = tf.where(condition=s_mask_processed, x=encoded_support,
                               y=tf.constant(-1e9, shape=(1, max_len, conv_dim)))
    # BNK, max len, conv dim
    another_rep_for_encoded_support = encoded_support * cos_similarity
    another_rep_for_encoded_support = tf.where(condition=s_mask_processed, x=another_rep_for_encoded_support,
                                               y=tf.constant(-1e9, shape=(1, max_len, conv_dim)))

    another_rep_for_encoded_support = tf.reduce_max(another_rep_for_encoded_support, axis=1)
    # BNK, conv dim
    encoded_support = tf.reduce_max(encoded_support, axis=1)
    # BNK, conv dim
    mix = tf.stack([another_rep_for_encoded_support, encoded_support], axis=-1)
    # BNK, 2 * conv dim
    encoded_support = Dense(units=1, kernel_initializer=keras.initializers.Constant([0.0, 1.0]))(mix)
    encoded_support = LabelContrastiveLoss([encoded_support, class_input])
    q_sentence = tf.reshape(q_input, (-1, max_len, word_embedding_dim))
    encoded_query = shared_conv(q_sentence)
    # [B * N * Q, hidden_size]
    q_mask_processed = tf.reshape(q_mask, shape=(-1, max_len))
    q_mask_processed = tf.cast(tf.tile(tf.expand_dims(q_mask_processed, axis=-1), multiples=[1, 1, conv_dim]), bool)
    encoded_query = tf.where(condition=q_mask_processed, x=encoded_query,
                             y=tf.constant(-1e9, shape=(1, max_len, conv_dim)))
    encoded_query = tf.reduce_max(encoded_query, axis=1)
    # [B, N, K, hidden_size]
    encoded_support = tf.reshape(encoded_support, shape=(-1, N, K, conv_dim))
    # [B, N * Q, hidden_size]
    encoded_query = tf.reshape(encoded_query, shape=(-1, N * Q, conv_dim))
    # ------------------------------------------ feature-level attention ------------------------------------------
    # [B * N, 1, K, hidden_size]
    fea_att_score = tf.reshape(encoded_support, shape=(-1, K, conv_dim, 1))  # channel last
    # [B * N, 32, K, hidden_size]
    fea_att_score = Conv2D(filters=32, kernel_size=(K, 1), padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(
        fea_att_score)
    # [B * N, 64, K, hidden_size]
    fea_att_score = Conv2D(filters=64, kernel_size=(K, 1), padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(
        fea_att_score)
    fea_att_score = Dropout(rate=0.5)(fea_att_score)

    # [B * N, 1, 1, hidden_size]
    fea_att_score = Conv2D(filters=1, kernel_size=(K, 1), strides=(K, 1),
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(
        fea_att_score)
    # [B, 1, N, hidden_size]
    fea_att_score = tf.expand_dims(tf.reshape(fea_att_score, shape=(-1, N, conv_dim)), axis=1)

    # ------------------------------------------ instance-level attention ------------------------------------------
    encoded_support = tf.tile(tf.expand_dims(encoded_support, axis=1), multiples=[1, N * Q, 1, 1, 1])
    support_for_att = shared_dense(encoded_support)
    encoded_query_processed = tf.expand_dims(tf.expand_dims(encoded_query, axis=2), axis=3)
    encoded_query_processed = tf.tile(encoded_query_processed, multiples=[1, 1, N, K, 1])  # B, NQ, {N, K}, conv dim
    query_for_att = shared_dense(encoded_query_processed)
    # [B, NQ, N, K]
    ins_att_score = tf.reduce_sum(tf.tanh(support_for_att * query_for_att), axis=-1)
    ins_att_score = tf.nn.softmax(ins_att_score, axis=-1)
    # [B, NQ, N, hidden_size]
    ins_att_score = tf.tile(tf.expand_dims(ins_att_score, axis=4), multiples=[1, 1, 1, 1, conv_dim])
    support_proto = tf.reduce_sum(encoded_support * ins_att_score, axis=3)
    # ------------------------------------------- distance computation -------------------------------------------
    encoded_query = tf.expand_dims(encoded_query, axis=2)
    y = ((support_proto - encoded_query) ** 2) * fea_att_score
    y = - tf.reduce_sum(y, axis=3)
    y = tf.nn.softmax(y, axis=-1)
    y = tf.reshape(y, shape=(-1, N, Q, N))

    model_hatt = Model(inputs=[s_input, q_input, s_mask, q_mask, class_input], outputs=y)

    return model_hatt

