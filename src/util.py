import tensorflow as tf

def get_record_parser(config):
    def parse(example):
        features = tf.parse_single_example(example, 
                                          features={
                                              "c_image" : tf.FixedLenFeature([], tf.string),
                                              "r_image" : tf.FixedLenFeature([], tf.string),
                                              "img_info" : tf.FixedLenFeature([], tf.string),
                                              "end_width": tf.FixedLenFeature([], tf.int64),
                                              "id" : tf.FixedLenFeature([], tf.int64)
                                          })
        c_image = tf.decode_raw(features["c_image"], tf.int32)
        r_image = tf.decode_raw(features["r_image"], tf.int32)
        img_info = tf.decode_raw(features["img_info"], tf.int32)
        end_width = features['end_width']
        id_ = features['id']
        return c_image, r_image, img_info, end_width, id_
    return parse 

def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    dataset = dataset.batch(config.batch_size)
    return dataset
