from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from core import CNNBlock, WaveletUnpooling, ReflectionPadding2D

class WaveletDecoder(Model):
    
    def __init__(self):
        super().__init__()
        self._name = 'WaveletDecoder'
        
        ### Block 3
        self.block3_conv2d_1 = CNNBlock(256, 3, 'WD_block4_conv2d_1')
        self.block3_unpooling = WaveletUnpooling('WD_block4_unpooling')
        self.block3_conv2d_2 = CNNBlock(256, 3, 'WD_block3_conv2d_2')
        self.block3_conv2d_3 = CNNBlock(256, 3, 'WD_block3_conv2d_3')
        self.block3_conv2d_4 = CNNBlock(256, 3, 'WD_block3_conv2d_4')
        
        ### Block 2
        self.block2_conv2d_1 = CNNBlock(128, 3, 'WE_block2_conv2d_1')
        self.block2_unpooling = WaveletUnpooling('WE_block2_unpooling')
        self.block2_conv2d_2 = CNNBlock(128, 3, 'WE_block2_conv2d_2')
            
        ### Block 1
        self.block1_conv2d_1 = CNNBlock(64, 3, 'WE_block1_conv2d_1')
        self.block1_unpooling = WaveletUnpooling('WE_block1_unpooling')
        self.block1_conv2d_2 = CNNBlock(64, 3, 'WE_block1_conv2d_2')
        
        self.post_processing_padding = ReflectionPadding2D()
        self.post_processing_conv2d = Conv2D(3, 3, padding = 'valid')
        
    def call(self, inputs, skips, trainable = False):
        x = self.block3_conv2d_1(inputs)
        x = self.block3_unpooling([x, *skips['block3']])
        x = self.block3_conv2d_2(x)
        x = self.block3_conv2d_3(x)
        x = self.block3_conv2d_4(x)
        
        x = self.block2_conv2d_1(x)
        x = self.block2_unpooling([x, *skips['block2']])
        x = self.block2_conv2d_2(x)
        
        x = self.block1_conv2d_1(x)
        x = self.block1_unpooling([x, *skips['block1']])
        x = self.block1_conv2d_2(x)
        
        x = self.post_processing_padding(x)
        x = self.post_processing_conv2d(x)
        
        return x