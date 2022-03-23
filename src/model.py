import tensorflow as tf
from tensorflow.keras import Model
from wavelet_ae import WaveletAE
from vggish import VGGish
from vggish_preprocessing.preprocess_sound import preprocess_sound

import pickle as pk
from config import PCA_WEIGHTS_DIR


class iMuse(Model):
    def __init__(self):
        super().__init__()
        self._name = 'iMuse'
        self.trainable = False
        
        self.wa = WaveletAE()
        self.vggish = VGGish()

        self.pca = []

        for block in range(1, 5):
            self.pca.append(pk.load(open(str(PCA_WEIGHTS_DIR / f'pca_{block}_music.pkl'),'rb')))
        
if __name__ == '__main__':
    imuse = iMuse()