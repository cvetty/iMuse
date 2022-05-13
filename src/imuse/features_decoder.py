import tensorflow as tf
from layers import FeatureExtractorTranspose
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dropout, Conv1DTranspose, Reshape, Dense, Concatenate, Add, Input

from config import DROPOUT_RATE, KERNEL_INITIALIZER, REGULARIZER

class FeaturesDecoder(Model):
    def __init__(self, block_level = 1):
        super().__init__()
        self._name = f'FeaturesDecoder{block_level}'
        self.block_level = block_level

        self.corr_output_size = 2 ** (5 + self.block_level)

        ### Main Network Sampling
        self.sample_corr = Dense((self.corr_output_size // 8)**2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.sample_reshape = Reshape((self.corr_output_size // 8, self.corr_output_size // 8))

        self.corr_fe1 = Conv1DTranspose(self.corr_output_size // 8, 3, activation='relu', padding='same', kernel_initializer=KERNEL_INITIALIZER)
        self.corr_fe2 = FeatureExtractorTranspose(self.corr_output_size // 4, upsampling='conv')
        self.corr_fe3 = FeatureExtractorTranspose(self.corr_output_size // 2, upsampling='conv')
        self.corr_fe4 = FeatureExtractorTranspose(self.corr_output_size, upsampling='conv')
        self.corr_out = Conv1DTranspose(self.corr_output_size, 1)

        self.means_net = Sequential([
            Dense(self.corr_output_size // 8, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dense(self.corr_output_size // 4, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dropout(DROPOUT_RATE),
            Dense(self.corr_output_size // 4, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dropout(DROPOUT_RATE),
            Dense(self.corr_output_size // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dense(self.corr_output_size),
        ])

    def call(self, inputs):
        ### Main Network
        corr_sample = self.sample_corr(inputs)
        corr_sample = self.sample_reshape(corr_sample)

        corr = self.corr_fe1(corr_sample)
        corr = self.corr_fe2(corr)
        corr = self.corr_fe3(corr)
        corr = self.corr_fe4(corr)
        corr = self.corr_out(corr)

        means = self.means_net(inputs)

        return corr, means
