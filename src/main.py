import tensorflow as tf
import os
import argparse as _argparse
from PIL import Image
import numpy as np
import collections
import copy
import random
from tqdm import tqdm
from util import get_record_parser, get_batch_dataset
from model import Model

def train(config):
    print("Building Model ... ")
    print("Setting Batch system ... ")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle( 
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    model = Model(config, iterator,is_train=True)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(config.train_log_dir)
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        #sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        total_loss = 0.0
        
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        
        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            
            loss = sess.run( [model.loss ], feed_dict={handle: train_handle, model.total_loss : 0} )
            print(loss)
            total_loss += loss[0]
            '''
            if (global_step+1) % config.stack_batch == 0:
                print(total_loss)
                loss, train_op = sess.run([model.loss, model.train_op], 
                                          feed_dict={ handle : train_handle,
                                                      model.total_loss : total_loss })
                total_loss = 0
            '''                           
            '''
            loss, final_output, lab_img, enc_img, dec_img, res_img, fin_state, fin_output, emit_ta, img_info, end_width = sess.run( 
                [model.loss, model.final_output, model.label_img, model.enc_img_, model.dec_img_, model.res_img_, model.fin_state, model.fin_output, model.emit_ta, model.img_info, model.end_width], 
                feed_dict={handle: train_handle})
            
            print('========================')
            print('Image Width    :', img_info[2])
            print('Image Height   :', img_info[3])
            print('End Width      :', end_width)
            print('label    Image :', np.shape(lab_img))
            print('Encoding Image :', np.shape(enc_img))
            print('Decoding Image :', np.shape(dec_img))
            print('ResNet   Image :', np.shape(res_img))
            print('RNN Fin  State :', np.shape(fin_state))
            print('RNN Fin Output :', np.shape(fin_output))
            print('Emit_ta Output :', np.shape(emit_ta))
            print('Final   Output :', np.shape(final_output))
            print('Loss           :', loss)
            #print('Model Info :', img_info)
            #print(end_width)
            print('========================')
            '''
            
            
            