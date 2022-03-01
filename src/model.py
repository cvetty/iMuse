from tensorflow.keras import Model
from wavelet_encoder import WaveletEncoder
from wavelet_decoder import WaveletDecoder

class WaveletAE(Model):
    def __init__(self):
        super().__init__()
        self._name = 'WaveletAE'
        self.encoder = WaveletEncoder()
        self.decoder = WaveletDecoder()

    def call(self, inputs, trainable = False):
        x, skips = self.encoder(inputs)
        output = self.decoder(x, skips)

        return output