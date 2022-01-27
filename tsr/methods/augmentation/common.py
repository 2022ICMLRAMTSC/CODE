import tensorflow as tf


def resize_time_series(series, new_length, method='bilinear'):
    '''
    Resize a time series, as if it was a 1 dimensional image
    Args:
        series:
        new_length:
        method:

    Returns:

    '''
    image_like = tf.expand_dims(series, axis = 0)
    return tf.image.resize(images=image_like, size= [1, new_length], method=method)[0]


def pad_to_length(series, target_length):
    return tf.pad(series, [[0, max(target_length - series.shape[0],0) ], [0, 0]])


def cut_time_series(series, cut_start, cut_end, insert=None):

    input_length = series.shape[0]
    to_concat = [series[:cut_start], insert, series[cut_end:]]

    series = tf.concat([x for x in to_concat if not x is None], axis = 0)
    series = pad_to_length(series, input_length)
    return series

def check_proba(proba, shape = None):
    if shape is None:
        shape = [1]
    return tf.random.uniform(shape = shape) <= proba