import tensorflow as tf
from imuse import FeaturesMapperBlock
from utils import TensorBoardImage
from tensorflow.io import FixedLenFeature, parse_tensor, parse_single_example
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import argparse
from pathlib import Path
from datetime import datetime

from config import BATCH_SIZE, EPOCHS, DATA_OUTPUT_PCA_TRAIN_DIR, DATA_OUTPUT_PCA_TEST_DIR, DATA_OUTPUT_PCA_VAL_DIR

TRAIN_DIR = Path(DATA_OUTPUT_PCA_TRAIN_DIR)
VAL_DIR = Path(DATA_OUTPUT_PCA_VAL_DIR) / "0.tfrecord"
TEST_DIR = Path(DATA_OUTPUT_PCA_TEST_DIR) / "0.tfrecord"
AUTOTUNE = tf.data.AUTOTUNE

def get_features_description():
    description = {
        'img_path': FixedLenFeature([], tf.string),
        'music_path': FixedLenFeature([], tf.string),
        'music_global_stats': FixedLenFeature([], tf.string)
    }
    
    for i in range(1, 5):
        description[f'img_block{i}_corr'] = FixedLenFeature([], tf.string)
        description[f'img_block{i}_mean'] = FixedLenFeature([], tf.string)

        description[f'music_block{i}_corr'] = FixedLenFeature([], tf.string)
        description[f'music_block{i}_mean'] = FixedLenFeature([], tf.string)
    
    return description

features_description = get_features_description()

def _parse_function(record, block_level = 1):
    parsed_data = parse_single_example(record, features_description)

    corr_shape = [2**(block_level + 5), 2 ** (3 + block_level)]
    means_shape = [2**(block_level + 5),]

    music_global_stats = parse_tensor(parsed_data["music_global_stats"], tf.float16)
    music_global_stats = tf.ensure_shape(music_global_stats, (512,))

    img_corr = parse_tensor(parsed_data[f'img_block{block_level}_corr'], tf.float16)
    img_corr = tf.ensure_shape(img_corr, [corr_shape[0], corr_shape[0]])

    img_means = parse_tensor(parsed_data[f'img_block{block_level}_mean'], tf.float16)
    img_means = tf.ensure_shape(img_means, means_shape)

    music_corr = parse_tensor(parsed_data[f'music_block{block_level}_corr'], tf.float16)
    music_corr = tf.ensure_shape(music_corr, [1, corr_shape[0]])
    music_corr = tf.reshape(music_corr, (-1,))

    music_means = parse_tensor(parsed_data[f'music_block{block_level}_mean'], tf.float16)
    music_means = tf.ensure_shape(music_means, means_shape)
    
    return (music_corr, music_means, music_global_stats), (img_corr, img_means)


def main(config):
    train_ds, test_ds, val_ds = preprocess_dataset(config.block)
    
    print(f'BLOCK: {config.block}')
    feature_mapper = FeaturesMapperBlock(config.block)
    feature_mapper.build([(1, 2**(5+config.block1)), (1, 2**(5+config.block1)), (1, 512)])
    feature_mapper.compile(tf.keras.optimizers.Adam(1e-4))
    feature_mapper.load_weights(f'../checkpoints/block{config.block}/FM.007-0.3820-0.3379-1.5898-2.3702.h5')

    print('Train Evaluation:')
    feature_mapper.evaluate(train_ds)

    print('Val Evaluation:')
    feature_mapper.evaluate(val_ds)

    print('Test Evaluation:')
    feature_mapper.evaluate(test_ds)


def preprocess_dataset(block_level = 1):
    # Global shuffle on all files
    train_ds_files = tf.data.Dataset.list_files(str(TRAIN_DIR / '*'), shuffle=True, seed=4321)

    train_ds = tf.data.TFRecordDataset(train_ds_files, compression_type="GZIP")
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.map(lambda x: _parse_function(x, block_level), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.data.TFRecordDataset(str(TEST_DIR), compression_type="GZIP")
    test_ds = test_ds.map(lambda x: _parse_function(x, block_level), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.TFRecordDataset(str(VAL_DIR), compression_type="GZIP")
    val_ds = val_ds.map(lambda x: _parse_function(x, block_level), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    # val_ds = val_ds.repeat()
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, val_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    config = parser.parse_args()

    if config.block == 0:
        for block in range(1, 5):
            config.block = block;
            main(config)
    else:
        main(config)