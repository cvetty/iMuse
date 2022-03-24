import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from config import VGGISH_WEIGHTS_PATH
from utils import get_correlations, preprocess_feat


class VGGish(Model):
    def __init__(self):
        super().__init__()
        self._name = 'VGGish'
        self.trainable = False

        # Block 1
        self.conv2d_1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')
        self.pool_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')

        # Block 2
        self.conv2d_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')
        self.pool_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')

        # Block 3
        self.conv2d_3_1 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.conv2d_3_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.pool_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')

        # Block 4
        self.conv2d_4_1 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')
        self.conv2d_4_2 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')
        self.pool_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')
        
        self.load_weights(VGGISH_WEIGHTS_PATH).expect_partial()

        self.global_pool = GlobalAveragePooling2D()

    def call(self, inputs, encode_level = None):
        feature_maps = {
            'block1': None,
            'block2': None,
            'block3': None,
            'block4': None,
            'global_stats': None
        }

        x = self.conv2d_1(inputs)
        x = self.pool_1(x)
        feature_maps['block1'] = x

        if encode_level == 1:
            return x

        x = self.conv2d_2(x)
        x = self.pool_2(x)
        feature_maps['block2'] = x

        if encode_level == 2:
            return x, feature_maps

        x = self.conv2d_3_1(x)
        x = self.conv2d_3_2(x)
        x = self.pool_3(x)
        feature_maps['block3'] = x

        if encode_level == 3:
            return x, feature_maps

        x = self.conv2d_4_1(x)
        x = self.conv2d_4_2(x)
        x = self.pool_4(x)
        feature_maps['block4'] = x

        feature_maps['global_stats'] = self.global_pool(x)

        return x, feature_maps

    def get_style_correlations(self, inputs, blocks=['block1', 'block2', 'block3', 'block4'], ede=True, normalize=True):
        _, encoder_feat = self.call(inputs)
        correlations = []
        means = []

        def process_feat(feat):
            return get_correlations(feat, normalize=normalize)

        for block in blocks:
            corr = tf.map_fn(process_feat, encoder_feat[block])
            mean = tf.map_fn(lambda feat: preprocess_feat(feat, center=False)[1], encoder_feat[block])
            correlations.append(corr)
            means.append(mean)

        return correlations, means, encoder_feat['global_stats']
        