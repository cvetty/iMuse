import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

from layers import CNNBlock, WaveletPooling
from utils import wct

class WaveletEncoder(Model):
    def __init__(self):
        super().__init__()
        self._name = 'WaveletEncoder'
        self.trainable = False

        self.preprocessing_conv2d = Conv2D(3, 1, padding='valid')

        ### Block 1
        self.block1_conv2d_1 = CNNBlock(64, 3, 'WE_block1_conv2d_1')
        self.block1_conv2d_2 = CNNBlock(64, 3, 'WE_block1_conv2d_2')
        self.block1_pooling = WaveletPooling('WE_block1_pooling')

        ### Block 2
        self.block2_conv2d_1 = CNNBlock(128, 3, 'WE_block2_conv2d_1')
        self.block2_conv2d_2 = CNNBlock(128, 3, 'WE_block2_conv2d_2')
        self.block2_pooling = WaveletPooling('WE_block2_pooling')

        ### Block 3
        self.block3_conv2d_1 = CNNBlock(256, 3, 'WE_block3_conv2d_1')
        self.block3_conv2d_2 = CNNBlock(256, 3, 'WE_block3_conv2d_2')
        self.block3_conv2d_3 = CNNBlock(256, 3, 'WE_block3_conv2d_3')
        self.block3_conv2d_4 = CNNBlock(256, 3, 'WE_block3_conv2d_4')
        self.block3_pooling = WaveletPooling('WE_block3_pooling')

        ### Block 4
        self.block4_conv2d_1 = CNNBlock(512, 3, 'WE_block4_conv2d_1')

    def call(self, inputs, style_corr=None, style_means=None, style_mix_coeff = 1, style_boost_coeff = 1, trainable = False):
        wavelet_skips = {
            'block1': None,
            'block2': None,
            'block3': None,
        }

        features = {
            'block1': None,
            'block2': None,
            'block3': None
        }

        x = self.preprocessing_conv2d(inputs)

        x = self.block1_conv2d_1(x)
        x = self.block1_conv2d_2(x)
        LL_1, LH_1, HL_1, HH_1 = self.block1_pooling(x)
        wavelet_skips['block1'] = [LH_1, HL_1, HH_1, x]
        features['block1'] = LL_1

        if style_corr and style_means:
            LL_1 = tf.map_fn(
                lambda x: wct(x[0], x[1], x[2], style_mix_coeff, style_boost_coeff),
                (LL_1, style_corr['block1'], style_means['block1']),
                dtype=LL_1.dtype
            )

        x = self.block2_conv2d_1(LL_1)
        x = self.block2_conv2d_2(x)
        LL_2, LH_2, HL_2, HH_2 = self.block2_pooling(x)
        wavelet_skips['block2'] = [LH_2, HL_2, HH_2, x]
        features['block2'] = LL_2

        if style_corr and style_means:
            LL_2 = tf.map_fn(
                lambda x: wct(x[0], x[1], x[2], style_mix_coeff, style_boost_coeff),
                (LL_2, style_corr['block2'], style_means['block2']),
                dtype=LL_2.dtype
            )

        x = self.block3_conv2d_1(LL_2)
        x = self.block3_conv2d_2(x)
        x = self.block3_conv2d_3(x)
        x = self.block3_conv2d_4(x)
        LL_3, LH_3, HL_3, HH_3 = self.block3_pooling(x)
        wavelet_skips['block3'] = [LH_3, HL_3, HH_3, x]
        features['block3'] = LL_3

        if style_corr and style_means:
            LL_3 = tf.map_fn(
                lambda x: wct(x[0], x[1], x[2], style_mix_coeff, style_boost_coeff),
                (LL_3, style_corr['block3'], style_means['block3']),
                dtype=LL_3.dtype
            )

        x = self.block4_conv2d_1(LL_3)
        features['block4'] = x

        if style_corr and style_means:
            x = tf.map_fn(
                lambda x: wct(x[0], x[1], x[2], style_mix_coeff, style_boost_coeff),
                (x, style_corr['block4'], style_means['block4']),
                dtype=x.dtype
            )

        return x, wavelet_skips, features