import keras
from keras import Model
from keras.layers import Conv1D, Input, Dense
import tensorflow as tf


def make_model(N, K, Q, Em, max_len, word_embedding_dim=50, conv_dim=50, kernel_size=3):
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
    
    l2_norm = tf.math.reduce_euclidean_norm(class_name, axis=-1) * tf.math.reduce_euclidean_norm(s_input_processed, axis=-1)
    # print(l2_norm.shape)
    cos_similarity = dot_product / l2_norm
    # print(cos_similarity.shape)
    # cos_similarity = cos_similarity * s_mask  # B, N, K, max len

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
    beta_final = tf.squeeze(beta_final, axis=-1, name='before_where')

    beta_final = tf.stack([beta_final, cos_similarity], axis=-1)  # B, N, K, max len, 2
    beta_final = Dense(units=1)(beta_final)
    beta_final = tf.squeeze(beta_final, axis=-1)
    
    beta_final = tf.where(condition=tf.cast(s_mask, dtype=bool), x=beta_final,
                          y=tf.constant(value=-1e9, shape=(1, N, K, max_len)), name='where')  # B, N, K, max len
    beta_final = tf.nn.softmax(beta_final, axis=-1)  # B, N, K, max len

    proto_raw = tf.matmul(tf.expand_dims(beta_final, axis=-2), s_sentence)  # B, N, K, 1, conv dim
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



