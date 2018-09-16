import tensorflow as tf

class Model(object):
    
    def __init__(self, config, batch):
        #### Input Data 부분
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        
        self.c_image, self.r_image, self.img_info, self.end_width, self.id_ = batch.get_next()
        self.c_image = tf.squeeze(self.c_image)
        self.r_image = tf.squeeze(self.r_image)
        self.img_info = tf.squeeze(self.img_info)
        self.end_width = tf.squeeze(self.end_width)
        self.id_ = tf.squeeze(self.id_)
        
        self.c_image = tf.reshape(self.c_image, [self.img_info[3], self.img_info[2], 3])
        self.r_image = tf.reshape(self.r_image, [self.img_info[3], self.img_info[2], 3])
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        '''
        #### Cutting & Copy Data 부분
        enc_seq_len = tf.divide(self.end_width, self.config.block_size)
        enc_seq_len = tf.floor(enc_seq_len) # Encoding 부분은 버림한다.
        
        dec_seq_len = tf.divide( self.img_info[2] - self.end_width, self.config.block_size )
        dec_seq_len = tf.add(tf.ceil(dec_seq_len), 1) # Decoding 부분은 올림한다.
        
        enc_img = tf.slice(self.c_image, 
                           [0,0,0], 
                           [self.img_info[3], enc_seq_len * self.config.block_size, 3])
        dec_img = tf.slice(self.c_image, 
                           [0, self.end_width, 0], 
                           [self.img_info[3], dec_seq_len * self.config.block_size, 3 ])
        
        #### Feature Extracting 부분
        
        enc_img = tf.slic
        #### Partitioning Data 부분
        
        enc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden)
        enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
        outputs, dec_states = tf.nn.dynamic_rnn()
        
        #### Feature Extracting 부분
        
        
        
        #### Seq2Seq 부분 
        
        #### Deconvolution 부분.
        
        '''
