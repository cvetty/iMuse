import tensorflow as tf
from layers import Sampler_Z
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv1DTransposed, BatchNormalization, Reshape, Add, Flatten, Dense

from layers import FeatureExtractor


class FeaturesEncoderBlock1(Model):
    def __init__(self, latent_dims=16, extractor_filters=16):
        super().__init__()
        self._name = 'FeaturesEncoderBlock1'
        self.feature_extractor = FeatureExtractor(extractor_filters)
        self.sampler = Sampler_Z()
        self.latent_dims = latent_dims

        self.means_processor = Sequential([
            Reshape((1, -1)),
            Conv1D(64, 1, activation='relu'),
            Conv1D(128, 1, activation='relu'),
            Conv1D(128, 1, activation='relu'),
        ])

        # -> 32 * 32
        self.preprocessing_conv = Conv1D(32, 1, activation='relu')

        # -> 16 * 64
        self.conv_1_1 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv_1_2 = Conv1D(64, 3, activation='relu',
                               padding='same', strides=2)
        self.bn_1 = BatchNormalization()

        # -> 8 * 128
        self.conv_2_1 = Conv1D(128, 3, activation='relu', padding='same')
        self.add_2 = Add()
        self.conv_2_2 = Conv1D(128, 3, activation='relu', padding='same')

        self.conv_2_3 = Conv1D(128, 3, activation='relu',
                               padding='same', dilation_rate=2, strides=2)
        self.bn_2 = BatchNormalization()

        self.flatten = Flatten()

        self.dense_mean = Dense(self.latent_dims)
        self.dense_std = Dense(self.latent_dims)

    def call(self, inputs, means_vector):
        x = self.feature_extractor(inputs)
        means = self.means_processor(means_vector)

        x = self.preprocessing_conv(x)
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.bn_1(x)

        x = self.conv_2_1(x)
        x = self.add_2()(means)
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)
        x = self.bn_2(x)

        x = self.flatten(x)
        mu = self.dense_mean(x)
        sigma = self.dense_std(x)

        z_sample, sd = self.sampler((mu, sigma))

        return z_sample, mu, sd


class FeaturesDecoderBlock1(Model):
    def __init__(self, latent_dims=16, extractor_filters=16):
        super().__init__()
        self._name = 'FeaturesDecoderBlock1'

        self.latent_dims = latent_dims
        self.sample_dense = Dense(16 * 64)
        self.reshape = Reshape((16, 64))

        # -> 8 * 128
        self.conv_2_1 = Conv1DTransposed(
            128, 3, activation='relu', padding='same')
        self.conv_2_2 = Conv1DTransposed(
            128, 3, activation='relu', padding='same', strides=2)
        self.bn_2 = BatchNormalization()

        # -> 16 * 64
        self.conv_1_1 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv_1_2 = Conv1D(64, 3, activation='relu',
                               padding='same', strides=2)
        self.bn_1 = BatchNormalization()

        self.means_out = Conv1D(1, 1)
        self.means_reshape = Reshape((-1,))
        self.corr_out = Conv1D(4, 3, padding='same')

    def call(self, z_sample, means_vector):
        x = self.sample_dense(z_sample)
        x = self.reshape(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.bn_2(x)

        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.bn_1(x)

        corr = self.corr_out(x)
        means = self.means_out(x)
        means = self.means_reshape(means)

        return corr, means


class FeaturesMapperBlock1(Model):
    def __init__(self):
        super().__init__()
        self._name = 'FeaturesMapperBlock1'

        self.encoder = FeaturesEncoderBlock1()
        self.decoder = FeaturesDecoderBlock1()

    def call(self, inputs, means_vector):
        z_sample, mu, sd = self.encoder(inputs, means_vector)
        corr, means = self.decoder(z_sample)

        kl_divergence = -0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd), axis=1)
        kl_divergence = tf.math.reduce_mean(kl_divergence)
        self.add_loss(self.kl_weight * kl_divergence)

        return corr, means
