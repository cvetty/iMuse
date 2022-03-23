import pandas as pd
import tensorflow as tf
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split

from pathlib import Path
from tqdm import tqdm

from data_generator import DatasetGenerator
from config import CSV_DATA_PATH, DATA_OUTPUT_PCA_TRAIN_DIR, DATA_OUTPUT_PCA_TEST_DIR, DATA_OUTPUT_PCA_VAL_DIR, BATCH_SIZE

import warnings

from config import BATCH_SIZE
warnings.filterwarnings('ignore')

tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
tf.get_logger().setLevel('ERROR')

import pickle as pk


import sys


def generate_data():
    data = pd.read_csv(CSV_DATA_PATH, index_col=0)
    x_train, x_test, y_train, y_test = train_test_split(
        data.music, data.img, test_size=0.15, stratify=data.quadrant, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.40)

    train_ds = generate_ds(x_train, y_train)
    test_ds = generate_ds(x_test, y_test)
    val_ds = generate_ds(x_val, y_val)

    dataset_gen = DatasetGenerator()
    pca = []

    for block in range(1, 5):
        block_pca = get_trained_pca(train_ds, dataset_gen, block)
        pk.dump(block_pca[0], open(f'../weights/pca/pca_{block}_music.pkl','wb'))
        pk.dump(block_pca[1], open(f'../weights/pca/pca_{block}_images.pkl','wb'))
        pca.append(block_pca)
        
        print(f'BLOCK {block} PCA:\nMusic:')
        print(sum(block_pca[0].explained_variance_ratio_))

        print(f'BLOCK {block} PCA:\nImages:')
        print(sum(block_pca[1].explained_variance_ratio_))

    dataset_gen.pca = pca

    write_as_TFRecords(
        dataset=train_ds,
        target_dir=DATA_OUTPUT_PCA_TRAIN_DIR,
        batch_size=1024,
        datagen=dataset_gen
    )

    write_as_TFRecords(
        dataset=test_ds,
        target_dir=DATA_OUTPUT_PCA_TEST_DIR,
        batch_size=len(list(test_ds)),
        datagen=dataset_gen
    )

    write_as_TFRecords(
        dataset=val_ds,
        target_dir=DATA_OUTPUT_PCA_VAL_DIR,
        batch_size=len(list(val_ds)),
        datagen=dataset_gen
    )


def generate_ds(x, y):
    ds = pd.DataFrame({'x': x, 'y': y})
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(ds))

    return ds


def get_trained_pca(ds, datagen, block):
    ds = ds.batch(2**(5 + block) + 16, drop_remainder=True)

    ipca_images = IncrementalPCA(n_components=2**(5 + block), batch_size=64)
    ipca_music = IncrementalPCA(n_components=2**(5 + block), batch_size=64)

    def load_image(data):
        img_path = data[1].numpy().decode("utf-8")
        datagen.process_image(img_path)
        feat = datagen.style_corr[block - 1]
        feat = tf.reshape(feat, (-1,))

        return feat

    def load_music(data):
        music_path = data[0].numpy().decode("utf-8")
        datagen.process_music(music_path)
        feat = datagen.vggish_feat_corr[block - 1]
        feat = tf.reshape(feat, (-1,))

        return feat

    for batch in tqdm(ds):
        valid_paths = tf.map_fn(lambda x: Path(x[0].numpy().decode("utf-8")).is_file() and Path(x[1].numpy().decode("utf-8")).is_file(), batch, tf.bool)
        batch = tf.boolean_mask(batch, valid_paths)

        images = tf.map_fn(load_image, batch, tf.float16).numpy()
        music = tf.map_fn(load_music, batch, tf.float16).numpy()

        ipca_images.partial_fit(images)
        ipca_music.partial_fit(music)

    return ipca_music, ipca_images

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
