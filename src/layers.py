import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, ReLU, Conv2D, Conv1D, MaxPooling1D, BatchNormalization, Attention, Flatten, Add, Concatenate

from utils import _conv2d, _conv2d_transpose
import sys


### Wavelet AE ###
class WaveletPooling(Layer):
    """
    Wavelet Pooing Custom Layer
    """

    def __init__(self, name=''):
        super(WaveletPooling, self).__init__()
        self._name = name
        square_of_2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        L = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant(
                [[1, 1]], dtype=tf.float32))
        )
        H = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant(
                [[-1, 1]], dtype=tf.float32))
        )

        self.LL = tf.reshape(tf.math.multiply(
            tf.transpose(L), L), (1, 2, 2, 1))
        self.LH = tf.reshape(tf.math.multiply(
            tf.transpose(L), H), (1, 2, 2, 1))
        self.HL = tf.reshape(tf.math.multiply(
            tf.transpose(H), L), (1, 2, 2, 1))
        self.HH = tf.reshape(tf.math.multiply(
            tf.transpose(H), H), (1, 2, 2, 1))

    def call(self, inputs):
        LL, LH, HL, HH = self.repeat_filters(inputs.shape[-1])
        return [_conv2d(inputs, LL),
                _conv2d(inputs, LH),
                _conv2d(inputs, HL),
                _conv2d(inputs, HH)]

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0], input_shape[1] // 2,
            input_shape[2] // 2, input_shape[3]
        )

        return [shape, shape, shape, shape]

    def repeat_filters(self, repeats):
        # Can we optimize this?
        return [
            tf.transpose(tf.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0))
        ]


class WaveletUnpooling(Layer):
    """
    Wavelet Unpooing Custom Layer
    """

    def __init__(self, name):
        super(WaveletUnpooling, self).__init__()
        self._name = name
        square_of_2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        L = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant(
                [[1, 1]], dtype=tf.float32))
        )
        H = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant(
                [[-1, 1]], dtype=tf.float32))
        )

        self.LL = tf.reshape(tf.math.multiply(
            tf.transpose(L), L), (1, 2, 2, 1))
        self.LH = tf.reshape(tf.math.multiply(
            tf.transpose(L), H), (1, 2, 2, 1))
        self.HL = tf.reshape(tf.math.multiply(
            tf.transpose(H), L), (1, 2, 2, 1))
        self.HH = tf.reshape(tf.math.multiply(
            tf.transpose(H), H), (1, 2, 2, 1))

    def call(self, inputs):
        LL_in, LH_in, HL_in, HH_in, tensor_in = inputs
        LL, LH, HL, HH = self.repeat_filters(LL_in.shape[-1])
        out_shape = tf.shape(tensor_in)

        return _conv2d_transpose(LL_in, LL, output_shape=out_shape) + _conv2d_transpose(LH_in, LH,
                                                                                        output_shape=out_shape) + _conv2d_transpose(
            HL_in, HL, output_shape=out_shape) + _conv2d_transpose(HH_in, HH, output_shape=out_shape)

    def compute_output_shape(self, input_shape):
        _ip_shape = input_shape[0]
        shape = (
            _ip_shape[0],
            _ip_shape[1] * 2,
            _ip_shape[2] * 2,
            sum(ips[3] for ips in input_shape)
        )

        return shape

    def repeat_filters(self, repeats):
        # Can we optimize this?
        return [
            tf.transpose(tf.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0)),
        ]


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if s[1] == None:
            return (None, None, None, s[3])
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        print(config)
        return config


class CNNBlock(Layer):
    def __init__(self, filters, kernel, name):
        super(CNNBlock, self).__init__()
        self._name = name

        self.padding = ReflectionPadding2D()
        self.conv2d = Conv2D(filters, kernel, padding='valid')
        self.activation = ReLU()

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.conv2d(x)
        x = self.activation(x)

        return x

### IMuse Model ###
class ExtractorCNNBlock(Model):
    def __init__(self, filters, kernel):
        super().__init__()
        self.conv1 = Conv1D(filters, (kernel,),
                            activation='relu', padding='same')
        self.pool = MaxPooling1D()
        self.conv2 = Conv1D(filters * 2, (kernel,),
                            activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)

        return x


class FeatureExtractor(Model):
    def __init__(self, filters = 16):
        super().__init__()

        self.block1 = ExtractorCNNBlock(filters, 1)
        self.block2 = ExtractorCNNBlock(filters, 3)
        self.block3 = ExtractorCNNBlock(filters, 5)
        self.block4 = ExtractorCNNBlock(filters, 7)
        self.bn = BatchNormalization()
        self.flatten = Flatten()
        self.attention = Attention()

    def call(self, inputs, flatten=False):
        block1_enc = self.block1(inputs)
        block2_enc = self.block2(inputs)
        block3_enc = self.block3(inputs)
        block4_enc = self.block4(inputs)

        x = Add()([block1_enc, block2_enc, block3_enc, block4_enc])
        x = self.bn(x)

        att = self.attention([x, x])
        x = Concatenate()([x, att])

        if flatten:
            x = self.flatten(x)

        return x


class Sampler(Layer):
    def call(self, inputs):
        mu, log_variance = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        epsilon = tf.random.normal(shape=(batch, dim))
        sampled_point = mu + tf.exp(log_variance / 2) * epsilon

        return sampled_point