import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanAbsoluteError

from imuse.features_encoder import FeaturesEncoder
from imuse.features_decoder import FeaturesDecoder

from config import BATCH_SIZE


class FeaturesMapperBlock(Model):
    def __init__(self, block_level=1, kl_weight = 1 / 0.5 * BATCH_SIZE):
        super(FeaturesMapperBlock, self).__init__()
        self._name = 'FeaturesMapperBlock'
        self.block_level = block_level

        self.encoder = FeaturesEncoder(self.block_level)
        self.decoder = FeaturesDecoder(self.block_level)
        self.kl_weight = kl_weight
        self._calculate_mar_loss = MeanAbsoluteError()

    def call(self, inputs):
        z_sample, self.mu, self.log_variance, global_stats = self.encoder(inputs[0], inputs[1], inputs[2])
        corr, means = self.decoder(z_sample, global_stats)

        return corr, means

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            reconstruction_loss, corr_loss, means_loss = self._calculate_reconstruction_loss(y, y_pred)
            kl_loss = self._calculate_kl_loss()
            loss = self.kl_weight * kl_loss + reconstruction_loss

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))

        return {
            'kl_loss': kl_loss,
            'mar': reconstruction_loss,
            'loss': loss,
            'corr': corr_loss,
            'means': means_loss,
        }

    def test_step(self, data):
        x, y = data

        y_pred = self.call(x)
        reconstruction_loss, corr_loss, means_loss = self._calculate_reconstruction_loss(y, y_pred)
        kl_loss = self._calculate_kl_loss()
        loss = self.kl_weight * kl_loss + reconstruction_loss

        return {
            'kl_loss': kl_loss,
            'mar': reconstruction_loss,
            'loss': loss,
            'corr': corr_loss,
            'means': means_loss,
        }

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(
            y_target, y_predicted)

        kl_loss = self._calculate_kl_loss()
        combined_loss = self.kl_weight * kl_loss + reconstruction_loss

        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        corr_loss = self._calculate_mar_loss(y_target[0], y_predicted[0])
        means_loss = self._calculate_mar_loss(y_target[1], y_predicted[1])
        reconstruction_loss = 0.5 * (corr_loss + means_loss)

        return reconstruction_loss, corr_loss, means_loss

    def _calculate_kl_loss(self):
        kl_loss = -0.5 * tf.math.reduce_sum(1 + self.log_variance - tf.square(
            self.mu) - tf.exp(self.log_variance), axis=1)
        kl_loss = tf.math.reduce_mean(kl_loss)

        return kl_loss
