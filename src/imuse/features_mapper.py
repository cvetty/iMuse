import tensorflow as tf
from tensorflow.keras import Model

from imuse.features_encoder import FeaturesEncoder
from imuse.features_decoder import FeaturesDecoder

class FeaturesMapperBlock(Model):
    def __init__(self, block_level = 1, kl_weight=1):
        super().__init__()
        self._name = 'FeaturesMapperBlock'
        self.block_level = block_level

        self.encoder = FeaturesEncoder(self.block_level)
        self.decoder = FeaturesDecoder(self.block_level)
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_sample, mu, sd, global_stats = self.encoder(inputs[0], inputs[1], inputs[2])
        corr, vectors, means  = self.decoder(z_sample, global_stats)

        kl_divergence = -0.5 * tf.math.reduce_sum(1 + tf.math.log(
            tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd), axis=1)
        kl_divergence = tf.math.reduce_mean(kl_divergence)
        self.add_loss(lambda: self.kl_weight * kl_divergence)

        return corr, vectors, means