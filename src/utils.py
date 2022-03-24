import numpy as np
import tensorflow as tf
from tensorflow.nn import conv2d, conv2d_transpose
from tensorflow.io import read_file
from tensorflow.image import decode_image
from tensorflow.keras.callbacks import Callback

import pywt
from PIL import Image
from skimage.util import img_as_ubyte
import random
import io


def _conv2d(x, kernel):
    return conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')


def _conv2d_transpose(x, kernel, output_shape):
    return conv2d_transpose(
        x, kernel,
        output_shape=output_shape,
        strides=[1, 2, 2, 1],
        padding='SAME')


def load_image(image_path):
    image = read_file(image_path)
    image = decode_image(image)

    return image


def resize_image(img, max_size):
    aspect_ratio = img.shape[0] / img.shape[1]

    if (aspect_ratio > 1):
        return tf.image.resize(img, (max_size, int(max_size // aspect_ratio)))
    else:
        return tf.image.resize(img, (int(max_size * aspect_ratio), max_size))


def preprocess_image(img, max_size=512):
    img = img / 255

    return resize_image(img, max_size)


# WCT Related Function
def preprocess_feat(feat, center=False):
    feat = tf.reshape(feat, (-1, feat.shape[-1]))
    feat = tf.transpose(feat)

    feat_mean_raw = tf.math.reduce_mean(feat, 1)

    if center:
        feat_mean = tf.expand_dims(feat_mean_raw, 1)
        feat = tf.subtract(feat, feat_mean)

    return feat, feat_mean_raw


def center_feat(feat):
    feat_mean = tf.math.reduce_mean(feat, 1)
    feat_mean = tf.expand_dims(feat_mean, 1)

    return tf.subtract(feat, feat_mean)


def get_svd(feat, with_corr_matrix = False, beta = 1):
    if not with_corr_matrix:
        feat = center_feat(feat)
        feat = tf.matmul(feat, feat, transpose_b=True) / (feat.shape[1] - 1)
    else:
        feat *= beta
        
    feat = feat + tf.eye(feat.shape[0])

    return tf.linalg.svd(feat)


def get_style_correlation_transform(feat, beta=1):
    s_e, _, s_v = get_svd(feat, with_corr_matrix=True, beta=beta)
    # The inverse of the content singular values matrix operation (^-0.5)
    s_e = tf.pow(s_e, 0.5)

    EDE = tf.matmul(tf.matmul(s_v, tf.linalg.diag(s_e)), s_v, transpose_b=True)

    return EDE


def wct(content_feat_raw, style_corr, style_mean, alpha = 1, beta = 1):
    style_EDE = get_style_correlation_transform(style_corr, beta)
    content_feat, content_mean = preprocess_feat(content_feat_raw, center=True)

    c_e, _, c_v = get_svd(content_feat)
    c_e = tf.pow(c_e, -0.5)

    content_EDE = tf.matmul(
        tf.matmul(c_v, tf.linalg.diag(c_e)), c_v, transpose_b=True)
    content_whitened = tf.matmul(content_EDE, content_feat)

    final_out = tf.matmul(style_EDE, content_whitened)
    final_out = tf.add(final_out, tf.reshape(style_mean, (-1, 1)))
    final_out = tf.clip_by_value(
        final_out,
        tf.math.reduce_min(content_feat),
        tf.math.reduce_max(content_feat),
    )

    final_out = tf.reshape(tf.transpose(final_out), content_feat_raw.shape)
    final_out = alpha * final_out + (1 - alpha) * content_feat_raw

    return final_out


def sample_from_corr_matrix(sigma, num_features=None, eigenvalues=None, eigenvectors=None):
    data = tf.eye(sigma.shape[0], num_features)

    if not eigenvectors or not eigenvectors:
        eigenvalues, _, eigenvectors = tf.linalg.svd(sigma)
        eigenvalues = tf.linalg.diag(eigenvalues)

    data = tf.matmul(tf.matmul(eigenvectors, tf.sqrt(eigenvalues)), data)

    return data


# Data Generator Helpers
def _bytes_feature(value, raw_string=False):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy() if not raw_string else value]))


def normalized_wt_downsampling(x, wavelet, level):
    LL = pywt.wavedec2(x, wavelet, 'periodization', level)[0]
    LL = LL / np.abs(LL).max()

    return LL

import sys
def get_correlations(feature_map, normalize='standard'):
    feat, _ = preprocess_feat(feature_map, center=True)
    feat = tf.matmul(feat, feat, transpose_b=True) / (feat.shape[1] - 1)

    if normalize == 'standard':
        feat -= tf.reduce_mean(feat)
        feat /= tf.math.reduce_std(feat)
    elif normalize == 'min-max':
        feat -= tf.reduce_min(feat)
        feat /= tf.reduce_max(feat) - tf.reduce_min(feat)

    return feat


def per_channel_wd(img, level=1, wavelet='haar'):
    r, g, b = tf.unstack(img, axis=2)
    r = normalized_wt_downsampling(r, wavelet, level)
    g = normalized_wt_downsampling(g, wavelet, level)
    b = normalized_wt_downsampling(b, wavelet, level)

    return tf.stack([r, g, b], axis=2)


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    _, height, width, channel = tensor.shape
    tensor = tensor[0]
    tensor_normalized = tensor - tensor.min()
    tensor_normalized /= tensor_normalized.max()
    tensor_normalized = img_as_ubyte(tensor_normalized)
    tensor_squeezed = np.squeeze(tensor_normalized)
    
    image = Image.fromarray(tensor_squeezed)
    output = io.BytesIO()

    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    summary = tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )

    return summary

class TensorBoardImage(Callback):
    def __init__(self, log_dir, ds):
        super().__init__()
        self.log_dir = log_dir
        self.ds = ds

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10:
            return
        
        data = random.choice(self.ds)
        x = data[0]
        y = data[1]

        music_corr = x[0][0:1]
        music_means = x[1][0:1]
        music_global_stats = x[2][0:1]

        img_corr_real = y[0][0:1]
        img_corr_pred, _ = self.model([music_corr, music_means, music_global_stats])

        self._write_corr_plot(img_corr_real, 'true', epoch)
        self._write_corr_plot(img_corr_pred, 'predicted', epoch)

    def _write_corr_plot(self, corr, filename, epoch):
        corr = tf.squeeze(corr, 0)
        corr = tf.expand_dims(corr, 2)

        corr = corr + (tf.abs(tf.reduce_min(corr)))
        corr = corr / tf.reduce_max(corr)
        corr = tf.cast(corr * 255, tf.uint8)

        tf.keras.utils.save_img(
            f'{self.log_dir}{epoch}-{filename}.jpg', corr, scale=False
        )