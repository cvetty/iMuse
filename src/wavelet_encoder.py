from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

from core import CNNBlock, WaveletPooling

class WaveletEncoder(Model):
    def __init__(self):
        super().__init__()
        self._name = 'WaveletEncoder'
        
        self.preprocessing_conv2d = Conv2D(3, 1, padding = 'valid')
        
        ### Block 1
        self.block1_conv2d_1 = CNNBlock(64, 3, 'WE_block1_conv2d_1')
        self.block1_conv2d_2 = CNNBlock(64, 3, 'WE_block1_conv2d_2')
        self.block1_pooling = WaveletPooling('WE_block1_pooling')
        
        ### Block 2
        self.block2_conv2d_1 = CNNBlock(128, 3, 'WE_block2_conv2d_1')
        self.block2_conv2d_2 = CNNBlock(128, 3, 'WE_block2_conv2d_2')
        self.block2_pooling = WaveletPooling('WE_block2_pooling')
        
        ### Block 3
        self.block3_conv2d_1 = CNNBlock(256, 3, 'WE_block3_conv2d_1')
        self.block3_conv2d_2 = CNNBlock(256, 3, 'WE_block3_conv2d_2')
        self.block3_conv2d_3 = CNNBlock(256, 3, 'WE_block3_conv2d_3')
        self.block3_conv2d_4 = CNNBlock(256, 3, 'WE_block3_conv2d_4')
        self.block3_pooling = WaveletPooling('WE_block3_pooling')
        
        ### Block 4
        self.block4_conv2d_1 = CNNBlock(512, 3, 'WE_block4_conv2d_1')

    def call(self, inputs, trainable = True):
        wavelet_skips = {
            'block1': None,
            'block2': None,
            'block3': None,
        }
        
        x = self.preprocessing_conv2d(inputs)
        
        x = self.block1_conv2d_1(x)
        x = self.block1_conv2d_2(x)
        LL_1, LH_1, HL_1, HH_1 = self.block1_pooling(x)
        wavelet_skips['block1'] = [LH_1, HL_1, HH_1, x]
        
        x = self.block2_conv2d_1(LL_1)
        x = self.block2_conv2d_2(x)
        LL_2, LH_2, HL_2, HH_2 = self.block2_pooling(x)
        wavelet_skips['block2'] = [LH_2, HL_2, HH_2, x]
        
        x = self.block3_conv2d_1(LL_2)
        x = self.block3_conv2d_2(x)
        x = self.block3_conv2d_3(x)
        x = self.block3_conv2d_4(x)
        LL_3, LH_3, HL_3, HH_3 = self.block3_pooling(x)
        wavelet_skips['block3'] = [LH_3, HL_3, HH_3, x]
        
        x = self.block4_conv2d_1(LL_3)
        
        return x, wavelet_skips