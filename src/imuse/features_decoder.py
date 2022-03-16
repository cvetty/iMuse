import tensorflow as tf
from layers import FeatureExtractorTranspose
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Conv2D, Reshape, Add, Dense

from config import DROPOUT_RATE
import sys
class FeaturesDecoder(Model):
    def __init__(self, block_level = 1):
        super().__init__()
        self._name = f'FeaturesDecoder{block_level}'
        self.block_level = block_level

        self.corr_output_shape = (2 ** (5 + self.block_level), 2 ** (1 + self.block_level), 1)
        self.first_layer_size = (self.corr_output_shape[0] // 2 if block_level < 3 else 64) * 2**2
        self.sample_size = (self.corr_output_shape[0] // 4) * (self.corr_output_shape[1] // 4) * self.first_layer_size
        self.sample_shape = (self.corr_output_shape[0] // 4, self.corr_output_shape[1] // 4, self.first_layer_size)

        ### Means Network
        self.sample_means = Dense(self.first_layer_size, activation='relu')
        self.means_gs_add = Add()
        self.means_dropout = Dropout(DROPOUT_RATE)
        self.means_dense = Dense(self.first_layer_size, activation='relu')
        self.means_out = Dense(self.corr_output_shape[0])

        ### Main Network Sampling
        self.sample_corr = Dense(self.sample_size, activation='relu')
        self.sample_reshape = Reshape(self.sample_shape)
        self.corr_gs_add = Add()

        ### Correlation Matrix Network + Vectors (PCA Transformed)
        self.correlations_net = self._get_decoder_cnn_net('corr')
        self.vectors_net = self._get_decoder_cnn_net('vec')

    def call(self, inputs, global_stats):
        ### Means Generation
        corr_means = self.sample_means(inputs)
        corr_means = self.means_gs_add([corr_means, global_stats])
        corr_means = self.means_dropout(corr_means)
        corr_means = self.means_dense(corr_means)
        corr_means = self.means_out(corr_means)

        ### Main Network
        corr_sample = self.sample_corr(inputs)
        corr_sample = self.sample_reshape(corr_sample)
        corr_sample = self.corr_gs_add([corr_sample, global_stats])

        corr_values = self.correlations_net(corr_sample)
        corr_vectors = self.vectors_net(corr_sample)

        corr = tf.matmul(corr_values, corr_vectors)

        return corr, corr_means

    def _get_decoder_cnn_net(self, out):
        in_shape = self.sample_shape if out == 'corr' else [self.sample_shape[1], self.sample_shape[0], self.sample_shape[2]]
        out_shape = self.corr_output_shape if out == 'corr' else [self.corr_output_shape[1], self.corr_output_shape[0], self.corr_output_shape[2]]
        
        return Sequential(
            [
                Input(self.sample_shape),
                Reshape(in_shape),

                FeatureExtractorTranspose(self.first_layer_size // 2, upsampling='conv'),
                FeatureExtractorTranspose(self.first_layer_size // 2**2, upsampling='conv', levels=1),

                Conv2D(1, 1),
                Reshape(out_shape[:2]),
            ]
        )