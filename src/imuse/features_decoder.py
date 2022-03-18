import tensorflow as tf
from layers import FeatureExtractorTranspose
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Conv2D, Reshape, Add, Dense, Conv2DTranspose, UpSampling2D

from config import DROPOUT_RATE, KERNEL_INITIALIZER, REGULARIZER
import sys

class FeaturesDecoder(Model):
    def __init__(self, block_level = 1):
        super().__init__()
        self._name = f'FeaturesDecoder{block_level}'
        self.block_level = block_level

        ### e.g. 64*64*1
        self.corr_output_size = 2 ** (5 + self.block_level)

        ### e.g. 64*64*1
        self.first_layer_size = self.corr_output_size // 4

        self.sample_shape = (self.corr_output_size // 8, self.corr_output_size // 8, self.first_layer_size * 2)
        self.sample_size = (self.sample_shape[0]**2 * self.sample_shape[-1])

        self.latent_dims = self.corr_output_size // 16

        ### Means Network
        self.sample_means = Dense(
            self.first_layer_size * 2**2,
            activation='relu',
            kernel_initializer=KERNEL_INITIALIZER,
        )
        self.means_gs_add = Add()
        self.means_dropout = Dropout(DROPOUT_RATE)
        self.means_dense = Dense(
            self.first_layer_size * 2**2,
            activation='relu',
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=REGULARIZER
        )

        self.means_out = Dense(self.corr_output_size)

        ### Main Network Sampling
        self.preprocessing_higher_dims = Dense(self.latent_dims // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.sample_corr = Dense(self.sample_size, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.sample_reshape = Reshape(self.sample_shape)
        self.corr_fe1 = FeatureExtractorTranspose(self.first_layer_size * 2**2, upsampling='conv', levels=1)
        self.corr_gs_reshape = Reshape((1, 1, -1))
        self.corr_gs_add = Add()
        self.corr_fe2 = FeatureExtractorTranspose(self.first_layer_size * 2, upsampling='conv', levels=1)
        self.corr_fe3 = FeatureExtractorTranspose(self.first_layer_size, upsampling='conv', levels=1)
        self.corr_out = Conv2D(1, 1)
        self.corr_out_reshape = Reshape((self.corr_output_size, self.corr_output_size))

    def call(self, inputs, global_stats):
        ### Means Generation
        corr_means = self.sample_means(inputs)
        corr_means = self.means_gs_add([corr_means, global_stats])
        corr_means = self.means_dropout(corr_means)
        corr_means = self.means_dense(corr_means)
        corr_means = self.means_out(corr_means)

        ### Main Network
        inputs = self.preprocessing_higher_dims(inputs)
        corr_sample = self.sample_corr(inputs)
        corr_sample = self.sample_reshape(corr_sample)

        corr_sample = self.corr_fe1(corr_sample)
        corr_gs_reshaped = self.corr_gs_reshape(global_stats)
        corr_sample = self.corr_gs_add([corr_sample, corr_gs_reshaped])

        corr = self.corr_fe2(corr_sample)
        corr = self.corr_fe3(corr)
        corr = self.corr_out(corr)
        corr = self.corr_out_reshape(corr)

        return corr, corr_means