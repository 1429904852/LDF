from keras.layers import Layer, Conv1D, ReLU, Embedding
import keras


class CNNEncoder(Layer):
    def __init__(self, conv_dim, kernel_size):
        super(CNNEncoder, self).__init__()
        self.conv = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1))

    def call(self, inputs, **kwargs):
        encoded = self.conv(inputs)
        return encoded


class Embedding_CNNEncoder(Layer):
    def __init__(self, word_2_vec_matrix, word_embedding_dim, conv_dim, kernel_size):
        super(Embedding_CNNEncoder, self).__init__()
        self.embed = Embedding(input_dim=len(word_2_vec_matrix), output_dim=word_embedding_dim,
                               embeddings_initializer=keras.initializers.Constant(word_2_vec_matrix), trainable=True,
                               mask_zero=True)
        self.conv = Conv1D(filters=conv_dim, kernel_size=kernel_size, padding='same',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                           bias_initializer=keras.initializers.RandomNormal(stddev=0.1))
        self.activation = ReLU()

    def call(self, inputs, **kwargs):
        encoded = self.embed(inputs)
        encoded = self.conv(encoded)
        encoded = self.activation(encoded)
        return encoded
