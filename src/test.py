from msilib.schema import Feature
from imuse import FeaturesMapperBlock

test = FeaturesMapperBlock(block_level=4)
test.build([(1, 512, 32), (1, 512), (1, 512)])
print(test.summary())