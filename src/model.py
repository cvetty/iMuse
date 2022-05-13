import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from tqdm import tqdm

import tensorflow as tf
from data_generator import DatasetGenerator
from imuse.features_mapper import FeaturesMapperBlock
from tensorflow.keras import Model
from wavelet_ae import WaveletAE
from vggish import VGGish
from vggish_preprocessing import vggish_params

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
        self.video_codec = VideoWriter_fourcc(*'mp4v')

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
            
            z_sample, mu, sigma = self.feature_mappers[block].encoder(raw_corr, raw_means, raw_global_stats)
            style_latent_coords[f'block{block + 1}'] = {
                'sample': z_sample,
                'mean': mu,
                'std': sigma
            }

        return style_latent_coords

    def animate(self, music_path, image, output_path, style_mix_coeff = 1, style_boost_coeff = 1, max_duration = 60, frame_length = 512, animation_type = 'stale'):
        music, sr = lb.load(music_path, duration=max_duration)
        music = lb.resample(music, sr, vggish_params.SAMPLE_RATE)
        music_len = music.shape[0] / vggish_params.SAMPLE_RATE
        spec = lb.feature.melspectrogram(y=music, sr=vggish_params.SAMPLE_RATE, n_mels=128,fmax=8000, hop_length=frame_length)

        specm = np.mean(spec,axis=0)
        gradm = np.gradient(specm)
        gradm = gradm.clip(min=0)

        def smooth(y, points = 9, beta = 25):
            box = np.kaiser(points, beta)
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth

        gradm = smooth(gradm, 24, 10)
        gradm /= np.max(gradm)
        gradm = tf.convert_to_tensor(gradm, tf.float32)

        specm = smooth(specm, 30, 5)
        specm /= np.max(specm)
        specm = tf.convert_to_tensor(specm, tf.float32)
        # plt.plot(gradm)
        # plt.plot(specm)
        # plt.show()
        # sys.exit()

        style_latent_data = self.get_latent_space_coords(music_path)
        last_grad = gradm[0]
        
        target_directions = {}

        FPS = int(gradm.shape[0] / music_len)
        video = VideoWriter(output_path, self.video_codec, float(FPS), (image.shape[2], image.shape[1]))

        for i, (spec_energy, grad) in tqdm(enumerate(zip(specm, gradm))):
            style_corrs = {}
            style_means = {}

            if (grad <= gradm[i-1] and grad < gradm[i+1 if i+1 < len(gradm) else 0]) or len(list(target_directions)) == 0:
                directions_set = len(list(target_directions)) != 0
                print('Changing direction')
                ### Generate new direction

                for block in range(4):
                    if directions_set:
                        target_direction = tf.random.normal((2 ** (block + 2),), -tf.random.uniform([]) * tf.reduce_mean(target_directions[f'block{block + 1}']))
                    else:
                        target_direction = tf.random.normal((2 ** (block + 2),), tf.random.uniform([], minval=-15, maxval=15))
                    
                    target_directions[f'block{block + 1}'] = target_direction
                    target_directions[f'block{block + 1}'] /= tf.norm(target_directions[f'block{block + 1}'])

            for block in range(4):
                if animation_type == 'stale':
                    block_latent_data = (1-grad) * style_latent_data[f'block{block + 1}']['sample'] + grad * style_boost_coeff * target_directions[f'block{block + 1}']
                elif animation_type == 'move':
                    block_latent_data = style_latent_data[f'block{block + 1}']['sample'] + ((8 * spec_energy) * grad) * target_directions[f'block{block + 1}']
                    style_latent_data[f'block{block + 1}']['sample'] = block_latent_data

                block_z_sample = tf.expand_dims(block_latent_data, 0)
                # block_z_sample = tf.random.normal((1, 2**(block + 2)))
                style_corrs[f'block{block + 1}'], style_means[f'block{block + 1}'] = self.feature_mappers[block].decoder(block_z_sample)
                
            ## Final Frame
            frame = self.wa.transfer(image, style_corrs, style_means, style_mix_coeff, grad * style_boost_coeff)[0]
            frame = (frame * 255).numpy().astype(np.uint8)

            video.write(frame)

            last_grad = last_grad

        video.release()

        video_clip = VideoFileClip(output_path)
        audio_clip = AudioFileClip(music_path)
        audio_clip = audio_clip.subclip(0, video_clip.end)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile('./t2.mp4')

        
if __name__ == '__main__':
    test_img = tf.io.read_file(r"D:\Projects\SCHOOL\IMuse\samples\content\swan1.jpg")
    test_img = tf.image.decode_image(test_img, dtype=tf.float16)
    test_img = tf.expand_dims(test_img, 0)
    imuse = iMuse()

    imuse.animate(r"D:\Projects\SCHOOL\IMuse\src\test1.mp3", test_img, './t35.mp4', 1, 40)
    
    # r = imuse.transfer(r"D:\Projects\SCHOOL\IMuse\src\MT0002399275.mp3", test_img, 1, 40)[0]
    # r = (r * 255).numpy().astype(np.uint8)
    # plt.imsave('./test_std-116ss.png', r)

    # test_img_style = tf.io.read_file(r"D:\Projects\SCHOOL\IMuse\assets\moli_style.jpg")
    # test_img_style = tf.image.decode_image(test_img_style, dtype=tf.float16)
    # test_img_style = tf.image.resize(test_img_style, (test_img_style.shape[0] // 4, test_img_style.shape[1] // 4))
    # r = imuse.wa.transfer_images(tf.expand_dims(test_img, 0), tf.expand_dims(test_img_style, 0), skips_transfer=False, decoder_transfer=False)[0]
    
    # r = (r * 255).numpy().astype(np.uint8)
    # plt.imsave('./test_std-16.png', r)
    # plt.imsave('test.jpg')