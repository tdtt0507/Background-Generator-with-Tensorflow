import tensorflow as tf
import tensorflow.contrib.slim as slim
from func import conv, bn, stack, _max_pool, activation, deconv2d, binary_crossentropy
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
        self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
        self.total_loss = tf.placeholder(tf.float32, [])
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

        print('Deconv Model area Building ...')
        self.deconv()
        print('Deconv Model area Builded')
        
        print('loss compute area Building ...')
        self.loss_compute()
        print('loss compute area Builded')
        
        if self.is_train:
            print('Learning Area Building ...')
            self.learning_area()
            print('Learning Area Builded')
        
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

        label_img = tf.slice(self.r_image, [0,0,0], [seq_len_height*self.block_size, self.img_info[2], 3])
        self.label_img = label_img
        
        
        
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
            
            enc_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.hidden_size)
            if self.is_train:
                enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.7)

            self.res_img = tf.reshape(self.res_img, [self.seq_len_height, self.enc_seq_len_width, 8192])
            enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, self.res_img, dtype=tf.float32)
            self.fin_state = enc_state
            self.fin_output = enc_outputs
        
        with tf.variable_scope('Decode'):
            
            dec_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.hidden_size)
            if self.is_train:
                dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.7)

            sequence_length = tf.cast(self.dec_seq_len_width, tf.int32)
            dummy_zero_input = tf.zeros(shape=[self.seq_len_height, self.config.hidden_size], dtype=tf.float32)

            
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
            
    def deconv(self):
        self.emit_ta= tf.reshape(self.emit_ta, [self.seq_len_height* self.dec_seq_len_width,8,8,128])
        self.emit_ta= deconv2d(self.emit_ta, output_dim=64, name="decon2d_1_img")
        self.emit_ta= deconv2d(self.emit_ta, output_dim=3, name="decon2d_2_img")
        self.emit_ta= tf.reshape(self.emit_ta, [self.seq_len_height, self.dec_seq_len_width, 32, 32, 3])
        
        self.output = tf.reshape(self.emit_ta, [self.seq_len_height * 32, -1, 3]) 
        self.final_output = tf.concat([self.emit_ta, self.enc_img_],1)
        self.final_output = tf.reshape( self.final_output, [self.seq_len_height * 32, -1, 3])
        
    def loss_compute(self):
        x_image = tf.slice(self.output, [0,0,0], [self.seq_len_height, self.img_info[2] - self.end_width, 3])
        y_image = tf.slice(self.r_image, [0,self.end_width,0],[self.seq_len_height, self.img_info[2] - self.end_width, 3])
        y_image = tf.image.convert_image_dtype(y_image, dtype=tf.float32)
        
        cross_entropy_L = binary_crossentropy(x_image, y_image)
        self.loss = tf.reduce_sum(cross_entropy_L)
        # 아직 Sequence Length 도 구현 안된 상태임.
        
    def learning_area(self):
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, self.config.grad_clip)
        
        self.train_op = self.opt.apply_gradients(
            zip(capped_grads, variables), global_step=self.global_step)
        