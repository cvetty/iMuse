from re import S
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb

import tensorflow as tf
from data_generator import DatasetGenerator
from imuse.features_mapper import FeaturesMapperBlock
from tensorflow.keras import Model
from wavelet_ae import WaveletAE
from vggish import VGGish
from vggish_preprocessing.preprocess_sound import preprocess_sound

import pickle as pk
from pathlib import Path
from config import PCA_WEIGHTS_DIR

import sys

class iMuse(Model):
    def __init__(self):
        super().__init__()
        self._name = 'iMuse'
        self.trainable = False
        
        self.wa = WaveletAE()
        self.vggish = VGGish()

        self.pca = []
        self.feature_mappers = []


        for block in range(1, 5):
            self.pca.append(pk.load(open(str(PCA_WEIGHTS_DIR / f'pca_{block}_music.pkl'),'rb')))
            self.feature_mappers.append(FeaturesMapperBlock(block_level=block, load_weights=True))

        self.datagen = DatasetGenerator(pca=self.pca)

    def transfer(self, music_path, image, style_mix_coeff = 1, style_boost_coeff = 1):
        style_corrs = {}
        style_means = {}

        ### VGGish Preprocessing
        self.datagen.process_music(music_path)

        ### VGGish -> Styles Features Generation
        for block in range(4):
            raw_corr = self.datagen.vggish_feat_corr[block]
            raw_means = tf.expand_dims(self.datagen.vggish_feat_means[block], 0)
            raw_global_stats = tf.expand_dims(self.datagen.vggish_global_stats, 0)
            
            style_data = self.feature_mappers[block]([raw_corr, raw_means, raw_global_stats])
            style_corrs[f'block{block + 1}'] = style_data[0]
            style_means[f'block{block + 1}'] = style_data[1]

        ## Transfer
        out = self.wa.transfer(image, style_corrs, style_means, style_mix_coeff, style_boost_coeff)

        return out

    def get_latent_space_coords(self, music_path):
        style_latent_coords = {}

        ### VGGish Preprocessing
        self.datagen.process_music(music_path)

        ### VGGish -> Styles Features Generation
        for block in range(4):
            raw_corr = self.datagen.vggish_feat_corr[block]
            raw_means = tf.expand_dims(self.datagen.vggish_feat_means[block], 0)
            raw_global_stats = tf.expand_dims(self.datagen.vggish_global_stats, 0)
            
            z_sample, mu, log_variance = self.feature_mappers[block].encoder(raw_corr, raw_means, raw_global_stats)
            style_latent_coords[f'block{block + 1}'] = {
                'z_sample': z_sample,
                'mu': mu,
                'sigma': log_variance
            }

        return style_latent_coords

    def animate(self, music_path, image, style_mix_coeff = 1, style_boost_coeff = 1, max_duration = 30, frame_length = 512, output_dir = Path('./')):
        music, sr = lb.load(music_path, duration=max_duration)

        spec = lb.feature.melspectrogram(y=music, sr=sr, n_mels=128,fmax=8000, hop_length=frame_length)

        specm = np.mean(spec,axis=0)
        gradm = np.gradient(specm)
        gradm = gradm/np.max(gradm)
        gradm = gradm.clip(min=0)

        ### gradm -> maps to style_boost_coeff

        specm = (specm-np.min(specm))/np.ptp(specm)

        frames_count = int(np.floor(max_duration * sr / frame_length))

        style_latent_data = self.get_latent_space_coords(music_path)

        
if __name__ == '__main__':
    test_img = tf.io.read_file('../assets/moli_content.jpg')
    test_img = tf.image.decode_image(test_img, dtype=tf.float16)
    test_img = tf.expand_dims(test_img, 0)
    imuse = iMuse()
    
    r = imuse.transfer(r"D:\Projects\SCHOOL\IMuse\data\music\OSTs\Set2\017.mp3", test_img, 1, 40)[0]
    r = (r * 255).numpy().astype(np.uint8)
    plt.imsave('./test_std-116ss.png', r)

    # test_img_style = tf.io.read_file(r"D:\Projects\SCHOOL\IMuse\assets\moli_style.jpg")
    # test_img_style = tf.image.decode_image(test_img_style, dtype=tf.float16)
    # test_img_style = tf.image.resize(test_img_style, (test_img_style.shape[0] // 4, test_img_style.shape[1] // 4))
    # r = imuse.wa.transfer_images(tf.expand_dims(test_img, 0), tf.expand_dims(test_img_style, 0), skips_transfer=False, decoder_transfer=False)[0]
    
    # r = (r * 255).numpy().astype(np.uint8)
    # plt.imsave('./test_std-16.png', r)
    # plt.imsave('test.jpg')