import tensorflow as tf
from tensorflow.nn import conv2d, conv2d_transpose
from tensorflow.io import read_file
from tensorflow.image import decode_image


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


### WCT Related Function
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


def get_svd(feat):
    feat = center_feat(feat)
    feat = tf.matmul(feat, feat, transpose_b=True) / (feat.shape[1] - 1)
    feat = feat + tf.eye(feat.shape[0])

    return tf.linalg.svd(feat)


def get_style_correlation_transform(feat, return_mean=False):
    feat, mean = preprocess_feat(feat)
    s_e, _, s_v = get_svd(feat)
    # The inverse of the content singular values matrix operation (^-0.5)
    s_e = tf.pow(s_e, 0.5)

    EDE = tf.matmul(tf.matmul(s_v, tf.linalg.diag(s_e)), s_v, transpose_b=True)

    return (EDE, mean) if return_mean else EDE


def wct(content_feat_raw, style_feat_raw, alpha=1):
    style_EDE, style_mean = get_style_correlation_transform(style_feat_raw, return_mean=True)
    content_feat, content_mean = preprocess_feat(content_feat_raw, center=True)

    c_e, _, c_v = get_svd(content_feat)
    c_e = tf.pow(c_e, -0.5)

    content_EDE = tf.matmul(tf.matmul(c_v, tf.linalg.diag(c_e)), c_v, transpose_b=True)
    content_whitened = tf.matmul(content_EDE, content_feat)

    final_out = tf.matmul(style_EDE, content_whitened)
    final_out = tf.add(final_out, tf.expand_dims(style_mean, 1))
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
