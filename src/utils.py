def _conv2d(x, kernel):
    return tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')

def _conv2d_transpose(x, kernel, output_shape):
    return tf.nn.conv2d_transpose(
            x, kernel,
            output_shape=output_shape,
            strides=[1, 2, 2, 1],
            padding='SAME')