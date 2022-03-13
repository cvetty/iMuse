import tensorflow as tf
from tensorflow.keras import Model

from imuse.features_encoder import FeaturesEncoder
from imuse.features_decoder import FeaturesDecoder
import sys
class FeaturesMapperBlock(Model):
    def __init__(self, block_level = 1, kl_weight=1):
        super(FeaturesMapperBlock, self).__init__()
        self._name = 'FeaturesMapperBlock'
        self.block_level = block_level

        self.encoder = FeaturesEncoder(self.block_level)
        self.decoder = FeaturesDecoder(self.block_level)
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_sample, self.mu, self.log_variance, global_stats = self.encoder(inputs[0], inputs[1], inputs[2])
        corr, vectors, means  = self.decoder(z_sample, global_stats)
        
        return corr, vectors, means

    def compile(self, optimizer, loss = None):
        super(FeaturesMapperBlock, self).compile(optimizer, self._calculate_combined_loss)
        self.optimizer = optimizer
        self.loss = self._calculate_combined_loss

    def _calculate_combined_loss(self, y_target, y_predicted):
        # reconstruction_loss = self._calculate_mse(y_target, y_predicted)
        # kl_loss = self._calculate_kl_loss(y_target, y_predicted)

        # combined_loss = self.kl_weight * reconstruction_loss + kl_loss

        return self._calculate_kl_loss(y_target, y_predicted)

    def _calculate_mse(self, y_target, y_predicted):
        loss = tf.square(y_target[0] - y_predicted[0]) + tf.square(y_target[1] - y_predicted[1]) + tf.square(y_target[2] - y_predicted[2])
        # loss = tf.reduce_mean(loss, axis=1)
        
        return loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * tf.math.reduce_sum(1 + self.log_variance - tf.square(self.mu) - tf.exp(self.log_variance), axis=1)
        
        return kl_loss