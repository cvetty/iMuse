import numpy as np
import librosa as lb
from sklearn.decomposition import PCA

import tensorflow as tf

from tensorflow.io import serialize_tensor
from tensorflow.train import Example, Features

import sys

from wavelet_ae import WaveletAE
from utils import resize_image, _bytes_feature
from vggish_preprocessing.preprocess_sound import preprocess_sound

tf.config.run_functions_eagerly(False)


class DatasetGenerator:
    def __init__(self, max_dim_size=128, vgg_input_max_size=512, pca_compressing_coeff=16):
        self.max_dim_size = max_dim_size
        self.vgg_input_max_size = vgg_input_max_size
        self.wavelet_ae = WaveletAE()
        self.pca = PCA()
        self.pca_compressing_coeff = pca_compressing_coeff

    def load_image(self, img_path):
        self.img_path = img_path
        self.img_bytes = tf.io.read_file(img_path)
        self.img_raw = tf.image.decode_image(
            self.img_bytes, channels=3, dtype=tf.float32)
        self.img_raw = resize_image(self.img_raw, self.max_dim_size)

        # Other usefull features
#         dwt_level = tf.experimental.numpy.log2(tf.reduce_max(self.img_raw.shape) / (self.max_dim_size + ((tf.reduce_max(self.img_raw.shape) - tf.reduce_min(self.img_raw.shape)) / 2)))
#         dwt_level = tf.round(dwt_level)
#         dwt_level = tf.cast(dwt_level, tf.uint8)

#         self.img_resized = per_channel_wd(self.img_raw, dwt_level)
#         self.img_resized = tfa.image.gaussian_filter2d(self.img_resized, (6, 6), sigma=6e-1)
#         self.img_resized = tf.image.resize_with_crop_or_pad(self.img_resized, self.max_dim_size, self.max_dim_size)

    def load_music(self, music_path):
        audio_data, sr = lb.load(music_path)
        train_len = 10 * sr
        random_start = np.random.randint(audio_data.shape[0] - train_len)
        audio_data = audio_data[random_start: random_start + train_len]

        self.spec = preprocess_sound(audio_data, sr)

    def get_style_transormations(self):
        feat, _ = self.wavelet_ae.get_features(tf.expand_dims(self.img_raw, 0))
        self.style_corr, self.style_means = self.wavelet_ae.get_style_correlations(
            tf.expand_dims(self.img_raw, 0), ede=False)

        for i in range(len(self.style_corr)):
            self.style_corr[i] = tf.squeeze(self.style_corr[i], 0)
            self.style_corr[i] = tf.cast(self.style_corr[i], tf.float16)
            self.style_corr[i] = self.get_corr_pca(
                self.style_corr[i], self.style_corr[i].shape[1] // self.pca_compressing_coeff)

        for i in range(len(self.style_means)):
            self.style_means[i] = tf.squeeze(self.style_means[i], 0)
            self.style_means[i] = tf.cast(self.style_means[i], tf.float16)


    def get_corr_pca(self, corr, n_components):
        feat = self.pca.fit_transform(corr.numpy())
        orthonormal_vectors = self.pca.components_

        feat = tf.convert_to_tensor(feat, dtype=tf.float16)
        orthonormal_vectors = tf.convert_to_tensor(
            orthonormal_vectors, dtype=tf.float16)

        return feat[:, :n_components], orthonormal_vectors[:n_components, :]

    def process(self, music, img):
        self.load_image(img)
        self.load_music(music)
        self.get_style_transormations()

    def serialize_information(self):
        features = {
            'img_path': _bytes_feature(self.img_path.encode('utf-8'), raw_string=True),
            'music_spec': _bytes_feature(serialize_tensor(self.spec)),
        }

        for i in range(1, 5):
            features[f'block{i}_corr'] = _bytes_feature(
                serialize_tensor(self.style_corr[i-1][0]))
            features[f'block{i}_orthonormal_vectors'] = _bytes_feature(
                serialize_tensor(self.style_corr[i-1][1]))
            features[f'block{i}_mean'] = _bytes_feature(
                serialize_tensor(self.style_means[i-1]))

        return Example(
            features=Features(feature=features)
        ).SerializeToString()
