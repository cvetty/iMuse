import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, ReLU, MaxPool2D, Conv2D, Conv2DTranspose, BatchNormalization, Attention, Concatenate, UpSampling2D

from utils import _conv2d, _conv2d_transpose

from config import DROPOUT_RATE


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


class FeatureExtractor(Model):
    def __init__(self, filters, kernel_size = 3, pool = None, levels=2, dilation_rate = 1):
        super().__init__()
        self.pool = None
        self.levels = levels

        self.conv = Conv2D(filters, kernel_size, activation='relu', padding='same', strides=2 if pool == 'conv' and levels == 1 else 1, dilation_rate=dilation_rate)

        if pool == 'max' and self.levels > 1:
            self.pool = MaxPool2D()
        elif pool == 'conv' and self.levels > 1:
            self.pool = Conv2D(filters, kernel_size, activation='relu', padding='same', strides=2)

        self.bn = BatchNormalization()
        self.attention = Attention()
        self.concat = Concatenate()
        self.postprocessing_conv = Conv2D(filters, 1, activation='relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)

        if self.pool and self.levels > 1:
            x = self.pool(x)

        att = self.attention([x, x])
        x = self.concat([x, att])

        x = self.postprocessing_conv(x)
        
        return x


class FeatureExtractorTranspose(Model):
    def __init__(self, filters, kernel_size = 3, upsampling = None, levels = 2):
        super().__init__()
        self.upsampling = None
        self.levels = levels

        self.conv = Conv2DTranspose(filters, kernel_size, activation='relu', padding='same', strides=2 if upsampling == 'conv' and levels == 1 else 1)

        if upsampling == 'max' and levels > 1:
            self.upsampling = UpSampling2D()
        elif upsampling == 'conv' and levels > 1:
            self.upsampling = Conv2DTranspose(filters, kernel_size, activation='relu', padding='same', strides=2)

        self.bn = BatchNormalization()
        self.attention = Attention()
        self.concat = Concatenate()
        self.postprocessing_conv = Conv2DTranspose(filters, 1, activation='relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)

        if self.upsampling and self.levels > 1:
            x = self.upsampling(x)

        att = self.attention([x, x])
        x = self.concat([x, att])

        x = self.postprocessing_conv(x)
        
        return x

class Sampler(Layer):
    def call(self, inputs):
        mu, log_variance = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        epsilon = tf.random.normal(shape=(batch, dim))
        sampled_point = mu + tf.exp(log_variance / 2) * epsilon

        return sampled_point