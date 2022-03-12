import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from pathlib import Path
from tqdm import tqdm

from data_generator import DatasetGenerator
from config import CSV_DATA_PATH, DATA_OUTPUT_TRAIN_DIR, DATA_OUTPUT_TEST_DIR, DATA_OUTPUT_VAL_DIR

import warnings
warnings.filterwarnings('ignore')

tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
tf.get_logger().setLevel('ERROR')


def generate_data():
    data = pd.read_csv(CSV_DATA_PATH, index_col=0)
    x_train, x_test, y_train, y_test = train_test_split(
        data.music, data.img, test_size=0.08, stratify=data.quadrant, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.15)

    train_ds = generate_ds(x_train, y_train)
    test_ds = generate_ds(x_test, y_test)
    val_ds = generate_ds(x_val, y_val)

    dataset_gen = DatasetGenerator()

    write_as_TFRecords(
        dataset=train_ds,
        target_dir=DATA_OUTPUT_TRAIN_DIR,
        batch_size=1024,
        datagen=dataset_gen
    )

    write_as_TFRecords(
        dataset=test_ds,
        target_dir=DATA_OUTPUT_TEST_DIR,
        batch_size=len(list(test_ds)),
        datagen=dataset_gen
    )

    write_as_TFRecords(
        dataset=val_ds,
        target_dir=DATA_OUTPUT_TRAIN_DIR,
        batch_size=len(list(val_ds)),
        datagen=dataset_gen
    )


def generate_ds(x, y):
    ds = pd.DataFrame({'x': x, 'y': y})
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(ds))

    return ds


def write_as_TFRecords(dataset, target_dir, batch_size, datagen):
    dataset = dataset.batch(batch_size)

    for part_id, data in enumerate(dataset):
        filename = str(Path(target_dir) / f"{part_id}.tfrecord")
        with tf.io.TFRecordWriter(filename, options=tf_record_options) as writer:
            for music, image in tqdm(data):
                music_path = music.numpy().decode("utf-8")
                image_path = image.numpy().decode("utf-8")
                if not Path(music_path).is_file() or not Path(image_path).is_file():
                    continue
                datagen.process(music_path, image_path)
                writer.write(datagen.serialize_information())
            writer.close()


if __name__ == '__main__':
    generate_data()
