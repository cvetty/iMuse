import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

from layers import CNNBlock, WaveletUnpooling, ReflectionPadding2D
from utils import wct

class WaveletDecoder(Model):
    def __init__(self):
        super().__init__()
        self._name = 'WaveletDecoder'
        self.trainable = False

        ### Block 3
        self.block3_conv2d_1 = CNNBlock(256, 3, 'WD_block4_conv2d_1')
        self.block3_unpooling = WaveletUnpooling('WD_block4_unpooling')
        self.block3_conv2d_2 = CNNBlock(256, 3, 'WD_block3_conv2d_2')
        self.block3_conv2d_3 = CNNBlock(256, 3, 'WD_block3_conv2d_3')
        self.block3_conv2d_4 = CNNBlock(256, 3, 'WD_block3_conv2d_4')

        ### Block 2
        self.block2_conv2d_1 = CNNBlock(128, 3, 'WE_block2_conv2d_1')
        self.block2_unpooling = WaveletUnpooling('WE_block2_unpooling')
        self.block2_conv2d_2 = CNNBlock(128, 3, 'WE_block2_conv2d_2')

        ### Block 1
        self.block1_conv2d_1 = CNNBlock(64, 3, 'WE_block1_conv2d_1')
        self.block1_unpooling = WaveletUnpooling('WE_block1_unpooling')
        self.block1_conv2d_2 = CNNBlock(64, 3, 'WE_block1_conv2d_2')

        self.post_processing_padding = ReflectionPadding2D()
        self.post_processing_conv2d = Conv2D(3, 3, padding='valid')

    def call(self, inputs, skips, style_feat=None, stylization_coeff=1, trainable=False):
        features = {
            'block3': None,
            'block2': None,
            'block1': None,
        }

        x = self.block3_conv2d_1(inputs)
        x = self.block3_unpooling([x, *skips['block3']])
        x = self.block3_conv2d_2(x)
        x = self.block3_conv2d_3(x)
        x = self.block3_conv2d_4(x)
        features['block3'] = x

        if style_feat:
            x = tf.map_fn(
                lambda x: wct(x[0], x[1], stylization_coeff),
                (x, style_feat['block3']),
                dtype=x.dtype
            )

        x = self.block2_conv2d_1(x)
        x = self.block2_unpooling([x, *skips['block2']])
        x = self.block2_conv2d_2(x)
        features['block2'] = x

        if style_feat:
            x = tf.map_fn(
                lambda x: wct(x[0], x[1], stylization_coeff),
                (x, style_feat['block2']),
                dtype=x.dtype
            )

        x = self.block1_conv2d_1(x)
        x = self.block1_unpooling([x, *skips['block1']])
        x = self.block1_conv2d_2(x)
        features['block1'] = x

        if style_feat:
            x = tf.map_fn(
                lambda x: wct(x[0], x[1], stylization_coeff),
                (x, style_feat['block1']),
                dtype=x.dtype
            )

        x = self.post_processing_padding(x)
        x = self.post_processing_conv2d(x)

        return x, features