from distutils.command.config import config
import tensorflow as tf
from imuse import FeaturesMapperBlock
from tensorflow.io import FixedLenFeature, parse_tensor, parse_single_example
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler

import argparse
from pathlib import Path
from datetime import datetime

from config import BATCH_SIZE, EPOCHS, DATA_OUTPUT_TRAIN_DIR, DATA_OUTPUT_TEST_DIR, DATA_OUTPUT_VAL_DIR

TRAIN_DIR = Path(DATA_OUTPUT_TRAIN_DIR)
VAL_DIR = Path(DATA_OUTPUT_VAL_DIR) / "0.tfrecord"
TEST_DIR = Path(DATA_OUTPUT_TEST_DIR) / "0.tfrecord"
AUTOTUNE = tf.data.AUTOTUNE

def get_features_description():
    description = {
        'img_path': FixedLenFeature([], tf.string),
        'music_path': FixedLenFeature([], tf.string),
        'music_global_stats': FixedLenFeature([], tf.string)
    }
    
    for i in range(1, 5):
        description[f'img_block{i}_corr'] = FixedLenFeature([], tf.string)
        description[f'img_block{i}_orthonormal_vectors'] = FixedLenFeature([], tf.string)
        description[f'img_block{i}_mean'] = FixedLenFeature([], tf.string)

        description[f'music_block{i}_corr'] = FixedLenFeature([], tf.string)
        description[f'music_block{i}_orthonormal_vectors'] = FixedLenFeature([], tf.string)
        description[f'music_block{i}_mean'] = FixedLenFeature([], tf.string)
    
    return description

features_description = get_features_description()

def _parse_function(record, block_level = 1):
    parsed_data = parse_single_example(record, features_description)

    corr_shape = [2**(block_level + 5), 2 ** (block_level + 1)]
    means_shape = [2**(block_level + 5),]

    music_global_stats = parse_tensor(parsed_data["music_global_stats"], tf.float16)
    music_global_stats = tf.ensure_shape(music_global_stats, (512,))

    img_corr = parse_tensor(parsed_data[f'img_block{block_level}_corr'], tf.float16)
    img_corr = tf.ensure_shape(img_corr, corr_shape)

    img_vectors = parse_tensor(parsed_data[f'img_block{block_level}_orthonormal_vectors'], tf.float16)
    img_vectors = tf.ensure_shape(img_vectors, corr_shape[::-1])

    img_means = parse_tensor(parsed_data[f'img_block{block_level}_mean'], tf.float16)
    img_means = tf.ensure_shape(img_means, means_shape)

    music_corr = parse_tensor(parsed_data[f'music_block{block_level}_corr'], tf.float16)
    music_corr = tf.ensure_shape(music_corr, corr_shape)

    music_means = parse_tensor(parsed_data[f'music_block{block_level}_mean'], tf.float16)
    music_means = tf.ensure_shape(music_means, means_shape)
    
    return (music_corr, music_means, music_global_stats), (img_corr, img_vectors, img_means)


def main(config):
    train_ds, test_ds, val_ds = preprocess_dataset(config.block)
    train_spe = 109
    test_spe = 12
    val_spe = 8
    
    feature_mapper = FeaturesMapperBlock(config.block)
    feature_mapper.compile(tf.keras.optimizers.Adam(5e-5))
    feature_mapper.fit(
        train_ds,
        steps_per_epoch = train_spe,
        validation_data = val_ds,
        validation_steps = val_spe,
        epochs=config.epochs,
        callbacks=get_callbacks(train_spe / 5, config)
    )


def get_callbacks(tensorboard_fq, config):
    log_dir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    return [
        TensorBoard(log_dir=log_dir, update_freq = tensorboard_fq),
        # EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 10, verbose = 1),
        ModelCheckpoint(
            filepath = f'../checkpoints/FeaturesMapper{config.block}/{{epoch:03d}}-{{loss:.4f}}-{{val_loss:.4f}}.h5',
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            save_best_only= True,
            verbose = 1
        ),
    ]


def preprocess_dataset(block_level = 1):
    # Global shuffle on all files
    train_ds_files = tf.data.Dataset.list_files(str(TRAIN_DIR / '*'), shuffle=True, seed=4321)

    train_ds = tf.data.TFRecordDataset(train_ds_files, compression_type="GZIP")
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.map(lambda x: _parse_function(x, block_level), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.repeat()
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
    main(config)