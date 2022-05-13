from pathlib import Path

from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

BASE_DIR = Path(__file__).resolve().parent.parent

WAVELET_AE_WEIGHTS_PATH = str(BASE_DIR.joinpath('weights', 'wavelet_autoencoder'))
VGGISH_WEIGHTS_PATH = str(BASE_DIR.joinpath('weights', 'vggish'))
PCA_WEIGHTS_DIR = BASE_DIR.joinpath('weights', 'pca')
FEATURE_MAPPERS_WEIGHTS_DIR = BASE_DIR.joinpath('weights', 'feature_mappers')


CSV_DATA_PATH = str(BASE_DIR.joinpath('data', 'music_images_data.csv'))
DATA_OUTPUT_TRAIN_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'train'))
DATA_OUTPUT_TEST_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'test'))
DATA_OUTPUT_VAL_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'val'))

DATA_OUTPUT_PCA_TRAIN_DIR = str(BASE_DIR.joinpath('data', 'tfrecords_pca', 'train'))
DATA_OUTPUT_PCA_TEST_DIR = str(BASE_DIR.joinpath('data', 'tfrecords_pca', 'test'))
DATA_OUTPUT_PCA_VAL_DIR = str(BASE_DIR.joinpath('data', 'tfrecords_pca', 'val'))

BATCH_SIZE = 32
EPOCHS = 1_200
DROPOUT_RATE = 0.3

KERNEL_INITIALIZER = HeNormal()
REGULARIZER = l2(l=5e-3)