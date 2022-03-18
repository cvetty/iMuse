from imuse import FeaturesMapperBlock
import tensorflow as tf
test = FeaturesMapperBlock(block_level=3)

test.build([(1, 256, 64), (1, 256), (1, 512)])
# print(test.encoder.summary())
# test.compile(optimizer='adam')
print(test.summary())
