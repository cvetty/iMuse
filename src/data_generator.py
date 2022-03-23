import numpy as np
import librosa as lb
from sklearn.decomposition import PCA

import tensorflow as tf
from vggish import VGGish

from tensorflow.io import serialize_tensor
from tensorflow.train import Example, Features

from wavelet_ae import WaveletAE
from utils import resize_image, _bytes_feature
from vggish_preprocessing.preprocess_sound import preprocess_sound

import sys

class DatasetGenerator:
    def __init__(self, max_dim_size=128, vgg_input_max_size=512, pca=None):
        self.max_dim_size = max_dim_size
        self.vgg_input_max_size = vgg_input_max_size
        self.wavelet_ae = WaveletAE()
        self.vggish = VGGish()
        self.pca = pca

    def process_image(self, img_path):
        self.img_path = img_path
        self.img_bytes = tf.io.read_file(img_path)
        self.img_raw = tf.image.decode_image(self.img_bytes, channels=3) / 255
        self.img_raw = resize_image(self.img_raw, self.max_dim_size)

        # Other usefull features
#         dwt_level = tf.experimental.numpy.log2(tf.reduce_max(self.img_raw.shape) / (self.max_dim_size + ((tf.reduce_max(self.img_raw.shape) - tf.reduce_min(self.img_raw.shape)) / 2)))
#         dwt_level = tf.round(dwt_level)
#         dwt_level = tf.cast(dwt_level, tf.uint8)

#         self.img_resized = per_channel_wd(self.img_raw, dwt_level)
#         self.img_resized = tfa.image.gaussian_filter2d(self.img_resized, (6, 6), sigma=6e-1)
#         self.img_resized = tf.image.resize_with_crop_or_pad(self.img_resized, self.max_dim_size, self.max_dim_size)

        self.style_corr, self.style_means = self.wavelet_ae.get_style_correlations(
            tf.expand_dims(self.img_raw, 0),
            normalize='standard',
            ede=False
        )

        if self.pca:
            self.style_corr_pca = []

        for i in range(len(self.style_corr)):
            self.style_corr[i] = tf.squeeze(self.style_corr[i], 0)
            self.style_corr[i] = tf.cast(self.style_corr[i], tf.float16)
            if self.pca:
                self.style_corr_pca.append(self._get_pca(self.pca[i][1], self.style_corr[i]))

        for i in range(len(self.style_means)):
            self.style_means[i] = tf.squeeze(self.style_means[i], 0)
            self.style_means[i] = tf.cast(self.style_means[i], tf.float16)

    def process_music(self, music_path):
        self.music_path = music_path
        audio_data, sr = lb.load(music_path)
        train_len = 10 * sr
        random_start = np.random.randint(audio_data.shape[0] - train_len)
        audio_data = audio_data[random_start: random_start + train_len]

        self.spec = preprocess_sound(audio_data, sr)
        self.vggish_feat_corr, self.vggish_feat_means, self.vggish_global_stats = self.vggish.get_style_correlations(
            tf.expand_dims(self.spec, 3),
            normalize='min-max',
            ede=False
        )
        self.vggish_global_stats = tf.squeeze(self.vggish_global_stats, 0)
        self.vggish_global_stats = tf.cast(self.vggish_global_stats, tf.float16)

        for i in range(len(self.vggish_feat_corr)):
            self.vggish_feat_corr[i] = tf.squeeze(self.vggish_feat_corr[i], 0)
            self.vggish_feat_corr[i] = tf.cast(self.vggish_feat_corr[i], tf.float16)
            if self.pca:
                self.vggish_feat_corr[i] = self._get_pca(self.pca[i][0], self.vggish_feat_corr[i])

        for i in range(len(self.vggish_feat_means)):
            self.vggish_feat_means[i] = tf.squeeze(self.vggish_feat_means[i], 0)
            self.vggish_feat_means[i] = tf.cast(self.vggish_feat_means[i], tf.float16)

    def _get_pca(self, pca, feat):
        feat = tf.reshape(feat, (1,-1))
        feat = pca.transform(feat.numpy())
        feat = tf.convert_to_tensor(feat, tf.float16)

        return feat

    def process(self, music, img):
        self.process_image(img)
        self.process_music(music)

    def serialize_information(self):
        features = {
            'img_path': _bytes_feature(self.img_path.encode('utf-8'), raw_string=True),
            'music_path': _bytes_feature(self.music_path.encode('utf-8'), raw_string=True),
            'music_global_stats': _bytes_feature(serialize_tensor(self.vggish_global_stats))
        }

        for i in range(1, 5):
            features[f'img_block{i}_corr'] = _bytes_feature(
                serialize_tensor(self.style_corr[i-1]))
            features[f'img_block{i}_mean'] = _bytes_feature(
                serialize_tensor(self.style_means[i-1]))

            if self.pca:
                features[f'img_block{i}_corr_pca'] = _bytes_feature(
                    serialize_tensor(self.style_corr_pca[i-1]))

            features[f'music_block{i}_corr'] = _bytes_feature(
                serialize_tensor(self.vggish_feat_corr[i-1]))
            features[f'music_block{i}_mean'] = _bytes_feature(
                serialize_tensor(self.vggish_feat_means[i-1]))

        return Example(
            features=Features(feature=features)
        ).SerializeToString()
t = DatasetGenerator()