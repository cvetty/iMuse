import tensorflow as tf
from layers import FeatureExtractorTranspose
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, Conv1DTranspose, BatchNormalization, Reshape, Add, Dense, Flatten

class FeaturesDecoder(Model):
    def __init__(self, block_level = 1):
        super().__init__()
        self._name = f'FeaturesDecoder{block_level}'

        self.sample_size = 2 ** (block_level + 2)
        self.first_layer_size = 2 ** ((block_level + 3 if block_level < 4 else 6) + 2)

        self.sample_dense = Dense(self.sample_size * self.first_layer_size)

        self.sample_reshape = Reshape((self.sample_size, -1))

        self.conv3 = Conv1DTranspose(self.first_layer_size // 2, 3, activation='relu', padding='same', strides=2)

        self.conv2 = Conv1DTranspose(self.first_layer_size // 2**2, 3, activation='relu', padding='same', strides=2)
        self.bn2 = BatchNormalization()

        self.feature_extractor = FeatureExtractorTranspose(self.first_layer_size // 2**3, levels=1)

        #in: self.conv1
        self.corr_values = Conv1DTranspose(self.first_layer_size // 2**4 if block_level != 4 else self.first_layer_size // 2**3, 1)

        self.corr_vectors_pool = MaxPooling1D()
        self.corr_vectors_bn = BatchNormalization()
        self.corr_vectors_conv = Conv1D(self.first_layer_size // 2 if block_level != 4 else self.first_layer_size, self.sample_size, padding='same', activation='relu', strides=2)
        self.corr_vectors_out = Conv1D(self.first_layer_size if block_level != 4 else self.first_layer_size * 2, 1)
        self.add = Add()

        self.means_conv1 = Conv1D(self.first_layer_size // 2**3, 3, padding='same', activation='relu', strides=2)
        self.mean_flatten = Flatten()
        self.means_dense1 = Dense(self.first_layer_size, activation='relu')
        self.means_dense2 = Dense(2**(block_level + 5))

    def call(self, inputs, global_stats):
        x = self.sample_dense(inputs)
        x = self.sample_reshape(x)
        x = self.add([x, global_stats])

        x = self.conv3(x)

        ### Orthonormal vectors generation from the middle of the network
        corr_vectors = self.corr_vectors_pool(x)
        corr_vectors = self.corr_vectors_bn(corr_vectors)
        corr_vectors = self.corr_vectors_conv(corr_vectors)
        corr_vectors = self.corr_vectors_out(corr_vectors)

        ### Means Generation Network
        corr_means = self.means_conv1(x)
        corr_means = self.mean_flatten(corr_means)
        corr_means = self.means_dense1(corr_means)
        corr_means = self.means_dense2(corr_means)

        ### The rest of the network is related to PCA's out of the CORR matrix
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.feature_extractor(x)

        corr_values = self.corr_values(x)
        corr = tf.matmul(corr_values, corr_vectors)

        return corr, corr_means