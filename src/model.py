import tensorflow as tf
from func import conv, bn, stack, _max_pool, activation
from rnn import my_dynamic_rnn

class Model(object):
    
    def __init__(self, config, batch, is_train):
        #### Input Data 부분
        print('Getting batch part Building ... ')
        self.is_train = is_train
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
        #self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        print('Getting batch part Builded')

        print('Cutting & Copy Part Building ... ')
        self.model_prepro()
        print('Cutting & Copy Part Builded')

        print('Resnet Model area Building ... ')
        self.small_resnet()
        print('Resnet Model area Builded')

        print('Seq2Seq Model area Building ...')
        self.conv_Seq2Seq()
        print('Seq2Seq Model area Builded')

    def model_prepro(self):

        #### Cutting & Copy Data 부분
        enc_seq_len_width = tf.divide(self.end_width, self.block_size)
        enc_seq_len_width = tf.cast(tf.floor(enc_seq_len_width), dtype=tf.int32) # Encoding 부분은 버림한다.

        seq_len_height = tf.divide(self.img_info[3], self.block_size)
        seq_len_height = tf.cast(tf.floor(seq_len_height), dtype=tf.int32)

        dec_seq_len_width = tf.divide( self.img_info[2] - self.end_width, self.block_size )
        dec_seq_len_width = tf.cast(tf.add(tf.ceil(dec_seq_len_width), 1), dtype=tf.int32) # Decoding 부분은 올림한다.
        # r_image 를 똑같이 자르는 것도 구현해야함. 

        enc_img = tf.slice(self.c_image, 
                           [0,0,0], 
                           [seq_len_height * self.block_size, enc_seq_len_width * self.block_size, 3])

        dec_img = tf.zeros(shape=[seq_len_height * dec_seq_len_width, self.block_size, self.block_size, 3], dtype=tf.float32)
        
        enc_img = tf.image.convert_image_dtype(enc_img, dtype=tf.float32)

        enc_img = tf.reshape(enc_img, [seq_len_height * enc_seq_len_width, self.block_size, self.block_size, 3])

        self.enc_seq_len_width = enc_seq_len_width
        self.seq_len_height = seq_len_height
        self.dec_seq_len_width = dec_seq_len_width

        self.enc_img_ = tf.reshape(enc_img, [seq_len_height, enc_seq_len_width, self.block_size, self.block_size, 3])
        self.dec_img_ = tf.reshape(dec_img, [seq_len_height, dec_seq_len_width, self.block_size, self.block_size, 3])
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
        self.res_img_ = tf.reshape(input_img, [self.seq_len_height, self.enc_seq_len_width, 8,8,128])

    def conv_Seq2Seq(self):

        with tf.variable_scope('Encode'):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
            if self.is_train:
                enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.7)

            self.res_img = tf.reshape(self.res_img, [self.seq_len_height, self.enc_seq_len_width, 8192])
            enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, self.res_img, dtype=tf.float32)
            self.fin_state = enc_state
            self.fin_output = enc_outputs
        
        with tf.variable_scope('Decode'):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
            if self.is_train:
                dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.7)
            # 입력값, Sequence 길이 : dec_seq_len_width
            # initial_state 또한 정의해줘야.
            sequence_length = tf.cast(self.dec_seq_len_width, tf.int32)
            dummy_zero_input = tf.zeros(shape=[self.seq_len_height, self.config.hidden_size], dtype=tf.float32)
            #output_ta = tf.TensorArray(size=self.seq_len_height, dtype=tf.int32)
            
            def dec_loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:
                    next_cell_state = enc_state
                    next_input = dummy_zero_input
                else:
                    next_cell_state = cell_state
                    next_input = cell_output
                
                elements_finished = (time >= sequence_length)
                finished = tf.reduce_all(elements_finished)
                next_loop_state = None
                
                return (elements_finished, next_input, next_cell_state,
                        emit_output, next_loop_state)
            
            emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(dec_cell, dec_loop_fn)
            
            self.emit_ta = emit_ta.stack()
            self.emit_ta = tf.transpose(self.emit_ta, [1,0,2])
            self.emit_ta = tf.reshape(self.emit_ta, [self.seq_len_height, self.dec_seq_len_width, 8, 8, 128])
            #self.dec_img = tf.reshape(self.dec_img, [self.seq_len_height, self.dec_seq_len_width, ])
        '''
        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
            if self.is_train:
                dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.7)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.zeros([8192], dtype=tf.float32), sequence_length=self.dec_seq_len_width)

            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state )

            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                impute_finished=True, maximum_iteration=20)
        self.res_img = outputs
        #with tf.variable_scope("rnn") as varscope:
        #    tf.get_variable("wi", shape=[512, self.config.hidden_size])

        #emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell = cell, loop_fn = crnn_loop_fn)
	'''
