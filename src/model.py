import tensorflow as tf
from func import conv, bn, stack, _max_pool, activation, crnn_loop_fn
class Model(object):
    
    def __init__(self, config, batch):
        #### Input Data 부분
        print('Getting batch part Building ... ')
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.block_size = tf.cast(self.config.block_size, dtype=tf.int32)
        self.c_image, self.r_image, self.img_info, self.end_width, self.id_ = batch.get_next()
        self.c_image = tf.squeeze(self.c_image)
        self.r_image = tf.squeeze(self.r_image)

        self.img_info = tf.squeeze(self.img_info)
        self.end_width = tf.cast( tf.squeeze(self.end_width), dtype=tf.int32)
        self.id_ = tf.squeeze(self.id_)
        
        self.c_image = tf.reshape(self.c_image, [self.img_info[3], self.img_info[2], 3])
        self.r_image = tf.reshape(self.r_image, [self.img_info[3], self.img_info[2], 3])
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        print('Getting batch part Builded')

        print('Cutting & Copy Part Building ... ')
        self.y_generator()
        print('Cutting & Copy Part Builded')

        print('Resnet Model area Building ... ')
        self.small_resnet()
        print('Resnet Model area Builded')

        print('Seq2Seq Model area Building ...')
        self.conv_Seq2Seq()
        print('Seq2Seq Model area Builded')

    def y_generator(self):

        #### Cutting & Copy Data 부분
        enc_seq_len = tf.divide(self.end_width, self.block_size)
        enc_seq_len = tf.cast(tf.floor(enc_seq_len), dtype=tf.int32) # Encoding 부분은 버림한다.

        dec_seq_len = tf.divide( self.img_info[2] - self.end_width, self.block_size )
        dec_seq_len = tf.cast(tf.add(tf.ceil(dec_seq_len), 1), dtype=tf.int32) # Decoding 부분은 올림한다.
        

        enc_img = tf.slice(self.c_image, 
                           [0,0,0], 
                           [self.img_info[3], enc_seq_len * self.block_size, 3])
        dec_img = tf.slice(self.c_image, 
                           [0, self.end_width, 0], 
                           [self.img_info[3], dec_seq_len * self.block_size, 3 ])
        dec_img = tf.zeros([self.img_info[3], dec_seq_len * self.block_size, 3], dtype=tf.int32)

        enc_img = tf.image.convert_image_dtype(enc_img, dtype=tf.float32)
        dec_img = tf.image.convert_image_dtype(dec_img, dtype=tf.float32)

        enc_img = tf.expand_dims(enc_img, 0)
        dec_img = tf.expand_dims(dec_img, 0)

        self.enc_img = enc_img
        self.dec_img = dec_img

        
    def small_resnet(self):

        tmp_config = dict()
        tmp_config['conv_filters_out'] = 64
        tmp_config['ksize'] = 5
        tmp_config['stride'] = 2
        tmp_config['use_bias'] = True
        input_img = self.enc_img
        with tf.variable_scope('scale1'):
            input_img = conv(input_img, tmp_config)
            input_img = bn(input_img, tmp_config)
            input_img = activation(input_img)

        with tf.variable_scope('scale2'):
            input_img = _max_pool(input_img, ksize=3, stride=2)
            tmp_config = dict()
            tmp_config['ksize_origin'] = 3
            tmp_config['num_blocks'] = 3
            tmp_config['stack_stride'] = 1
            tmp_config['block_filters_internal'] = 64
            tmp_config['use_bias'] = True
            input_img = stack(input_img, tmp_config)

        with tf.variable_scope('scale3'):
            tmp_config = dict()
            tmp_config['ksize_origin'] = 3
            tmp_config['num_blocks'] = 4
            tmp_config['stack_stride'] = 1
            tmp_config['block_filters_internal'] = 128
            tmp_config['use_bias'] = True
            input_img = stack(input_img, tmp_config)

        config = self.config

        self.res_img = input_img
        div = tf.cast(tf.floor( tf.divide(tf.shape(self.res_img)[2] , config.cut_size)), dtype=tf.int32)
        begin = tf.shape(self.res_img)[2] - (div * config.cut_size)
        self.res_img = tf.slice(self.res_img, [0,0, begin, 0], [1, tf.shape(self.res_img)[1], div * config.cut_size, 128])

        self.res_img = tf.reshape( self.res_img, [1, tf.shape(self.res_img)[1], div, config.cut_size * 128 ])
        self.res_img = tf.transpose(self.res_img, perm=[0,2,1,3])
        self.div = div 

    def conv_Seq2Seq(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(
                                                        cell = cell,
                                                        loop_fn = crnn_loop_fn)
