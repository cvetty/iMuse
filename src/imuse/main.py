import tensorflow as tf
from tensorflow.keras import Model

from vggish import VGGish
from wavelet_ae import WaveletAE

class IMuse(Model):
    def __init__(self):
        super().__init__()
        self._name = 'IMuse'

        self.vggish = VGGish()
        self.wavelet_ae = WaveletAE()
        