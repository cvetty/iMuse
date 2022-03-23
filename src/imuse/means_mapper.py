from tensorflow.keras.layers import Add, Dense, Dropout
from tensorflow.keras import Model


from config import KERNEL_INITIALIZER, REGULARIZER, DROPOUT_RATE

class MeansMapper(Model):
    def __init__(self, block_level=1):
        super(MeansMapper, self).__init__()
        self._name = 'MeansMapper'
        self.block_level = block_level

        self.means_dense1 = Dense(2**(5 + block_level), activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.means_dense2 = Dense(2**(5 + block_level), activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)
        self.dropout2 = Dropout(DROPOUT_RATE)

        self.means_dense3 = Dense(2**(5 + block_level), activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.dropout2 = Dropout(DROPOUT_RATE)
        self.means_dense4 = Dense(2**(5 + block_level), activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)
        
        self.means_out = Dense(2**(5+block_level))


        self.gs_dense1 = Dense(512, activation='relu', kernel_initializer=KERNEL_INITIALIZER)
        self.gs_dropout1 = Dropout(DROPOUT_RATE)
        self.gs_dense2 = Dense(2**(5 + block_level), activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)
        self.gs_add = Add()

    def call(self, inputs, gs):
        means = self.means_dense1(inputs)
        gs = self.gs_dense1(gs)
        gs = self.gs_dropout1(gs)
        gs = self.gs_dense2(gs)

        means = self.gs_add([means, gs])
        means = self.means_dense2(means)
        means = self.dropout2(means)
        means = self.means_dense3(means)
        means = self.means_out(means)

        return means