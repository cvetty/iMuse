from tensorflow.nn import conv2d, conv2d_transpose
from tensorflow.io import read_file
from tensorflow.image import decode_image

def _conv2d(x, kernel):
    return conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')

def _conv2d_transpose(x, kernel, output_shape):
    return conv2d_transpose(
            x, kernel,
            output_shape=output_shape,
            strides=[1, 2, 2, 1],
            padding='SAME')

def load_image(image_path):
    image = read_file(image_path)
    image = decode_image(image)

    return image