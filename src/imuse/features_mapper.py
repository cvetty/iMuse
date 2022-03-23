import tensorflow as tf
from imuse.means_mapper import MeansMapper
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

from imuse.features_encoder import FeaturesEncoder
from imuse.features_decoder import FeaturesDecoder

from config import BATCH_SIZE
import sys

class FeaturesMapperBlock(Model):
    def __init__(self, block_level=1, kl_weight = 1 / BATCH_SIZE):
        super(FeaturesMapperBlock, self).__init__()
        self._name = f'FeaturesMapper{block_level}'
        self.block_level = block_level

        self.encoder = FeaturesEncoder(self.block_level)
        self.decoder = FeaturesDecoder(self.block_level)
        self.kl_weight = kl_weight
        self._calculate_mar_loss = MeanSquaredError()

        self.means_mapper = MeansMapper(block_level)

    def call(self, inputs, training=False):
        z_sample, self.mu, self.log_variance = self.encoder(inputs[0], inputs[1], inputs[2])
        corr = self.decoder(z_sample)

        if not training:
            means = self.means_mapper(inputs[1], inputs[2])
        else:
            means = None

        return corr, means

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            ### MAIN
            corr_pred, _ = self.call(x, training = True)
            corr_mse_loss = self._calculate_mar_loss(y[0], corr_pred)
            kl_loss = self._calculate_kl_loss()
            corr_loss = self.kl_weight * kl_loss + corr_mse_loss

            corr_training_vars = self.trainable_variables
            corr_gradients = tape.gradient(corr_loss, corr_training_vars)

            self.optimizer.apply_gradients(zip(corr_gradients, corr_training_vars))

        with tf.GradientTape() as tape:
            ### MEANS
            means_pred = self.means_mapper(x[1], x[2])
            means_loss = self._calculate_mar_loss(y[1], means_pred)
            means_training_vars = self.means_mapper.trainable_variables
            means_gradients = tape.gradient(means_loss, means_training_vars)

            self.optimizer.apply_gradients(zip(means_gradients, means_training_vars))

        overall_loss = tf.abs(corr_loss) + tf.abs(means_loss)

        return {
            'kl_loss': kl_loss,
            'corr_mse': corr_mse_loss,

            'corr_loss': corr_loss,
            'means_loss': means_loss,
            'overall_loss': overall_loss
        }

    def test_step(self, data):
        x, y = data

        corr_pred, means_pred = self.call(x)
        corr_mse_loss = self._calculate_mar_loss(y[0], corr_pred)
        kl_loss = self._calculate_kl_loss()
        corr_loss = self.kl_weight * kl_loss + corr_mse_loss

        means_pred = self.means_mapper(x[1], x[2])
        means_loss = self._calculate_mar_loss(y[1], means_pred)
        overall_loss = tf.abs(corr_loss) + tf.abs(means_loss)


        return {
            'kl_loss': kl_loss,
            'corr_mse': corr_mse_loss,
            'corr_loss': corr_loss,

            'means_loss': means_loss,
            'overall_loss': overall_loss
        }


    def _calculate_kl_loss(self):
        kl_loss = -0.5 * tf.math.reduce_sum(1 + self.log_variance - tf.square(
            self.mu) - tf.exp(self.log_variance), axis=1)
        kl_loss = tf.math.reduce_mean(kl_loss)

        return kl_loss
