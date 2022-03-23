from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, Reshape, Add, Flatten, Dense, Concatenate, MaxPooling2D

from layers import Sampler, FeatureExtractor
from config import DROPOUT_RATE, KERNEL_INITIALIZER, REGULARIZER
import sys

class FeaturesEncoder(Model):
    def __init__(self, block_level=1):
        super().__init__()
        self._name = f'FeaturesEncoder{block_level}'

        self.block_level = block_level
        self.raw_input_shape = 2**(5 + block_level)
        self.latent_dims = self.raw_input_shape // 16
        
        self.dense1_1 = Dense(self.raw_input_shape, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.dense1_2 = Dense(self.raw_input_shape, activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(DROPOUT_RATE)

        self.dense2_1 = Dense(self.raw_input_shape // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.dense2_2 = Dense(self.raw_input_shape // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(DROPOUT_RATE)

        self.dense3_1 = Dense(self.raw_input_shape // 4, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.dense3_2 = Dense(self.raw_input_shape // 4, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.bn3 = BatchNormalization()
        self.dropout3 = Dropout(DROPOUT_RATE)

        self.dense4 = Dense(self.raw_input_shape // 8, activation='relu', kernel_initializer=KERNEL_INITIALIZER)

        self.add = Add()

        self.global_stats_processor = Sequential([
            Input((2**(block_level + 5) + 512,)),
            Dense(self.raw_input_shape // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dropout(DROPOUT_RATE),
            Dense(self.raw_input_shape // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
            Dropout(DROPOUT_RATE),
            Dense(self.raw_input_shape // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER),
        ])

        self.sampler = Sampler()
        self.dense_mean = Dense(self.latent_dims)
        self.dense_std = Dense(self.latent_dims)

        self.gs_concat = Concatenate()

        if self.block_level > 2:
            self.postprocessing = Dense(self.latent_dims // 2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)

    def call(self, inputs, means_vector, global_stats):
        global_stats = self.gs_concat([means_vector, global_stats])
        global_stats = self.global_stats_processor(global_stats)

        x = self.dense1_1(inputs)
        x = self.dense1_2(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.dense2_1(x)
        x = self.dense2_2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.add([x, global_stats])


        x = self.dense3_1(x)
        x = self.dense3_2(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.dense4(x)

        mu = self.dense_mean(x)
        log_variance = self.dense_std(x)

        z_sample = self.sampler((mu, log_variance))

        return z_sample, mu, log_variance