from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Reshape, Add, Flatten, Dense, Concatenate, MaxPooling2D

from layers import Sampler, FeatureExtractor


class FeaturesEncoder(Model):
    def __init__(self, block_level=1):
        super().__init__()
        self._name = f'FeaturesEncoder{block_level}'

        self.block_level = block_level
        self.raw_input_shape = (2 ** (5 + self.block_level), 2 ** (1 + self.block_level))
        self.preprocessing_reshape = Reshape([*self.raw_input_shape, 1])
        self.first_layer_size = self.raw_input_shape[0] / 2 if block_level < 3 else 64

        self.latent_dims = 16 if block_level in [1, 2] else 32

        self.preprocessing_conv = Conv2D(self.first_layer_size, 1, activation='relu', padding='same')
        self.feature_extractor1 = FeatureExtractor(self.first_layer_size, pool='max')
        self.feature_extractor2 = FeatureExtractor(self.first_layer_size * 2, pool='max')
        self.feature_extractor3 = FeatureExtractor(self.first_layer_size * 2**2, pool='conv', dilation_rate=2)
        self.pool = MaxPooling2D(padding='same')

        self.add = Add()

        self.global_stats_processor = Sequential([
            Input((2**(block_level + 5) + 512,)),
            Dense(self.first_layer_size * 2**2, activation='relu'),
            Dense(self.first_layer_size * 2**2, activation='relu'),
            Dense(self.first_layer_size * 2**2, activation='relu'),
        ])

        self.global_stats_reshape = Reshape((1, 1, -1))

        self.flatten = Flatten()

        self.sampler = Sampler()
        self.dense_mean = Dense(self.latent_dims)
        self.dense_std = Dense(self.latent_dims)

        self.gs_concat = Concatenate()

    def call(self, inputs, means_vector, global_stats):
        x = self.preprocessing_reshape(inputs)
        x = self.preprocessing_conv(x)
        x = self.feature_extractor1(x)
        x = self.feature_extractor2(x)
        x = self.feature_extractor3(x)
        x = self.pool(x)

        global_stats = self.gs_concat([means_vector, global_stats])
        global_stats = self.global_stats_processor(global_stats)
        global_stats_reshaped = self.global_stats_reshape(global_stats)

        x = self.add([x, global_stats_reshaped])

        x = self.flatten(x)

        mu = self.dense_mean(x)
        log_variance = self.dense_std(x)

        z_sample = self.sampler((mu, log_variance))

        return z_sample, mu, log_variance, global_stats
