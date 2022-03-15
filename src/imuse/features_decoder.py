import tensorflow as tf
from layers import FeatureExtractorTranspose
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Conv1DTranspose, BatchNormalization, Reshape, Add, Flatten, Dense

import sys
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
        self.bn = BatchNormalization()

        self.feature_extractor = FeatureExtractorTranspose(self.first_layer_size // 2**3, levels=1)

        #in: self.conv1
        self.corr_out = Conv1DTranspose(self.first_layer_size // 2**4 if block_level != 4 else self.first_layer_size // 2**3, 1)

        #in: self.conv1
        self.corr_means_out = Conv1D(1, 1)

        self.corr_vectors_pool1 = MaxPooling1D()
        self.corr_vectors_conv1 = Conv1D(self.first_layer_size // 2 if block_level != 4 else self.first_layer_size, self.sample_size, padding='same', activation='relu', strides=2)
        self.corr_vectors_out = Conv1D(self.first_layer_size if block_level != 4 else self.first_layer_size * 2, 1)
        self.add = Add()

    def call(self, inputs, global_stats):
        x = self.sample_dense(inputs)
        x = self.sample_reshape(x)
        x = self.add([x, global_stats])

        x = self.conv3(x)

        corr_vectors = self.corr_vectors_pool1(x)
        corr_vectors = self.corr_vectors_conv1(corr_vectors)
        corr_vectors = self.corr_vectors_out(corr_vectors)

        x = self.conv2(x)
        x = self.bn(x)
        
        x = self.feature_extractor(x)

        corr_out = self.corr_out(x)
        corr_means = self.corr_means_out(x)

        corr = tf.matmul(corr_out, corr_vectors)

        return corr, corr_means