from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, GlobalMaxPooling2D

class VGGish(Model):
    def __init__(self):
        super().__init__()
        self._name = 'VGGish'
        self.trainable = False

        # Block 1
        self.conv2d_1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')
        self.pool_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')

        # Block 2
        self.conv2d_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')
        self.pool_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')

        # Block 3
        self.conv2d_3_1 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.conv2d_3_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.pool_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')

        # Block 4
        self.conv2d_4_1 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')
        self.conv2d_4_2 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')
        self.pool_4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')

        self.global_pool = GlobalMaxPooling2D()

        self.load_weights('../weights/vggish')

    def call(self, inputs):
        skips = {
            'block1': None,
            'block2': None,
            'block3': None,
            'block4': None
        }

        x = self.conv2d_1(inputs)
        x = self.pool_1(x)
        skips['block1'] = x

        x = self.conv2d_2(x)
        x = self.pool_2(x)
        skips['block2'] = x

        x = self.conv2d_3_1(x)
        x = self.conv2d_3_2(x)
        x = self.pool_3(x)
        skips['block3'] = x

        x = self.conv2d_4_1(x)
        x = self.conv2d_4_2(x)
        x = self.pool_4(x)
        skips['block4'] = x

        x = self.global_pool(x)

        return x, skips