import multiprocessing as mp
import tensorflow as tf

from skimage.io import imread

CPU_CORES = mp.cpu_count()

def write_tfrecords(labels, file):
    with tf.io.TFRecordWriter(file) as tf_writer:
        pool = mp.Pool(CPU_CORES)
        for label in labels:
            pool.apply_async(to_tf_records_example,
                             args=[label],
                             callback=lambda example: tf_writer.write(example),
                             error_callback=lambda exception: print("error converting to tfrecords example: {}".format(exception)))
        pool.close()
        pool.join()

def to_tf_records_example(label):
    def _bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    
    
    def _float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    
    
    def _int64_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    name, (x, y) = label
    img = os.path.join(BUILD_IMG_DIR, name + ".png")
    img = imread(img, as_gray=True)
    assert img.shape == (406, 528)
    img = img.reshape([-1])  # flatten image into sequence of rows
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": _int64_feature([round(float(x)), round(float(y))]),
        "image": _int64_feature(img),
        "name": _bytes_feature([(name + ".png").encode("utf-8")])
    }))
    return example.SerializeToString()
