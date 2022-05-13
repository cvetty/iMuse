import tensorflow as tf
from imuse.means_mapper import MeansMapper
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

from imuse.features_encoder import FeaturesEncoder
from imuse.features_decoder import FeaturesDecoder

from config import BATCH_SIZE, FEATURE_MAPPERS_WEIGHTS_DIR
import sys

class FeaturesMapperBlock(Model):
    def __init__(self, block_level=1, kl_weight = 1 / BATCH_SIZE, load_weights=False):
        super(FeaturesMapperBlock, self).__init__()
        self._name = f'FeaturesMapper{block_level}'
        self.block_level = block_level

        self.encoder = FeaturesEncoder(self.block_level)
        self.decoder = FeaturesDecoder(self.block_level)
        self.kl_weight = kl_weight
        self._calculate_mar_loss = MeanSquaredError()

        if load_weights:
            self.build([(1, self.encoder.raw_input_shape,), (1, self.encoder.raw_input_shape,), (1, 512)])
            self.load_weights(str(FEATURE_MAPPERS_WEIGHTS_DIR / f'block{block_level}.h5'))

    def call(self, inputs, training=False):
        z_sample, self.mu, self.log_variance = self.encoder(inputs[0], inputs[1], inputs[2])
        corr, means = self.decoder(z_sample)

        return corr, means

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            ### MAIN
            corr_pred, means_pred = self.call(x, training = True)
            corr_mse_loss = self._calculate_mar_loss(y[0], corr_pred)
            means_mse_loss = self._calculate_mar_loss(y[1], means_pred)

            kl_loss = self._calculate_kl_loss()

            loss = (self.kl_weight * kl_loss) + (0.5 * (corr_mse_loss + means_mse_loss))

            corr_training_vars = self.trainable_variables
            corr_gradients = tape.gradient(loss, corr_training_vars)

            self.optimizer.apply_gradients(zip(corr_gradients, corr_training_vars))

        return {
            'kl_loss': kl_loss,
            'corr_mse': corr_mse_loss,
            'means_mse': means_mse_loss,
            'loss': loss,
        }

    def test_step(self, data):
        x, y = data

        corr_pred, means_pred = self.call(x)
        corr_mse_loss = self._calculate_mar_loss(y[0], corr_pred)
        means_mse_loss = self._calculate_mar_loss(y[1], means_pred)

        kl_loss = self._calculate_kl_loss()
        loss = (self.kl_weight * kl_loss) + (0.5 * (corr_mse_loss + means_mse_loss))


        return {
            'kl_loss': kl_loss,
            'corr_mse': corr_mse_loss,
            'means_mse': means_mse_loss,
            'loss': loss,
        }

    def _calculate_kl_loss(self):
        kl_loss = -0.5 * tf.math.reduce_sum(1 + self.log_variance - tf.square(
            self.mu) - tf.exp(self.log_variance), axis=1)
        kl_loss = tf.math.reduce_mean(kl_loss)

        return kl_loss

    def sample_latent_vector(self):
        return tf.random.normal((1, self.encoder.latent_dims))