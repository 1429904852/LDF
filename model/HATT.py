import keras
import tensorflow as tf
from keras.layers import Dense, Input, Conv2D, Dropout
from keras import Model
from model import Encoder


def make_model(N, K, Q, max_len, word_embedding_dim=50, conv_dim=50, kernel_size=3):
    shared_dense = Dense(units=conv_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=0.1))
    shared_conv = Encoder.CNNEncoder(conv_dim=conv_dim, kernel_size=kernel_size)

    s_input = Input(shape=(N, K, max_len, word_embedding_dim))
    q_input = Input(shape=(N, Q, max_len, word_embedding_dim))
    s_mask = Input(shape=(N, K, max_len))
    q_mask = Input(shape=(N, Q, max_len))

    # [B * N * K, seq_len, hidden_size]
    s_sentence = tf.reshape(s_input, (-1, max_len, word_embedding_dim))
    encoded_support = shared_conv(s_sentence)
    # [B * N * K, hidden_size]
    s_mask_processed = tf.reshape(s_mask, shape=(-1, max_len))
    s_mask_processed = tf.cast(tf.tile(tf.expand_dims(s_mask_processed, axis=-1), multiples=[1, 1, conv_dim]), bool)
    encoded_support = tf.where(condition=s_mask_processed, x=encoded_support,
                               y=tf.constant(-1e9, shape=(1, max_len, conv_dim)))
    # BNK, max len, conv dim
    encoded_support = tf.reduce_max(encoded_support, axis=1)
    # BNK, conv dim
    q_sentence = tf.reshape(q_input, (-1, max_len, word_embedding_dim))
    encoded_query = shared_conv(q_sentence)
    # [B * N * Q, hidden_size]
    q_mask_processed = tf.reshape(q_mask, shape=(-1, max_len))
    q_mask_processed = tf.cast(tf.tile(tf.expand_dims(q_mask_processed, axis=-1), multiples=[1, 1, conv_dim]), bool)
    encoded_query = tf.where(condition=q_mask_processed, x=encoded_query,
                             y=tf.constant(-1e9, shape=(1, max_len, conv_dim)))
    # B, N, K, max_len, conv_dim,  B, N, K, max_len
    encoded_query = tf.reduce_max(encoded_query, axis=1)
    # BNK, conv dim
    # [B, N, K, hidden_size]
    encoded_support = tf.reshape(encoded_support, shape=(-1, N, K, conv_dim))

    # [B, N * Q, hidden_size]
    encoded_query = tf.reshape(encoded_query, shape=(-1, N * Q, conv_dim))

    # ------------------------------------------ feature-level attention ------------------------------------------
    # [B * N, 1, K, hidden_size]
    fea_att_score = tf.reshape(encoded_support, shape=(-1, K, conv_dim, 1))  # channel last
    fea_att_score = Conv2D(filters=32, kernel_size=(K, 1), padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(fea_att_score)
    fea_att_score = Conv2D(filters=64, kernel_size=(K, 1), padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(fea_att_score)
    fea_att_score = Dropout(rate=0.5)(fea_att_score)

    fea_att_score = Conv2D(filters=1, kernel_size=(K, 1), strides=(K, 1),
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1), activation='relu')(fea_att_score)
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

    model_hatt = Model(inputs=[s_input, q_input, s_mask, q_mask], outputs=y)

    return model_hatt

