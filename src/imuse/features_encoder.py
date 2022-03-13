from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Reshape, Add, Flatten, Dense, Concatenate

from layers import Sampler, FeatureExtractor
import sys

class FeaturesEncoder(Model):
    def __init__(self, block_level = 1):
        super().__init__()
        self._name = f'FeaturesEncoder{block_level}'

        self.first_layer_size = 2 ** (block_level + 3 if block_level < 4 else 6)
        self.latent_dims = self.first_layer_size / 2

        self.feature_extractor = FeatureExtractor(self.first_layer_size // 2)

        self.preprocessing_conv = Conv1D(self.first_layer_size, 1, activation='relu')

        self.conv1 = Conv1D(self.first_layer_size * 2, 3, activation='relu', padding='same', strides=2)
        self.bn1 = BatchNormalization()

        self.conv2 = Conv1D(self.first_layer_size * 2**2, 3, activation='relu', padding='same', strides=2)
        self.bn2 = BatchNormalization()

        self.add = Add()
        self.global_stats_processor = Sequential([
            Input((2**(block_level + 5) + 512,)),
            Reshape((1, -1)),
            Conv1D(self.first_layer_size * 2**2, 1, activation='relu', padding='same'),
            Conv1D(self.first_layer_size * 2**2, 1, activation='relu', padding='same'),
            Conv1D(self.first_layer_size * 2**2, 1, activation='relu', padding='same'),
        ])

        self.flatten = Flatten()
        self.sampler = Sampler()
        self.dense_mean = Dense(self.latent_dims)
        self.dense_std = Dense(self.latent_dims)
        self.gs_concat = Concatenate()

    def call(self, inputs, means_vector, global_stats):
        x = self.feature_extractor(inputs)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        global_stats = self.gs_concat([means_vector, global_stats])
        global_stats = self.global_stats_processor(global_stats)

        x = self.add([x, global_stats])

        x = self.flatten(x)
        mu = self.dense_mean(x)
        log_variance = self.dense_std(x)

        z_sample = self.sampler((mu, log_variance))

        return z_sample, mu, log_variance, global_stats