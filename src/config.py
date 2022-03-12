from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_DATA_PATH = str(BASE_DIR.joinpath('data', 'music_images_data.csv'))
DATA_OUTPUT_TRAIN_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'train'))
DATA_OUTPUT_TEST_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'test'))
DATA_OUTPUT_VAL_DIR = str(BASE_DIR.joinpath('data', 'tfrecords', 'val'))
