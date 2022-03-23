from imuse import FeaturesMapperBlock
import tensorflow as tf
test = FeaturesMapperBlock(block_level=4)

test.build([(1, 512), (1, 512), (1, 512)])
# print(test.encoder.summary())
# test.compile(optimizer='adam')
print(test.encoder.summary())
    