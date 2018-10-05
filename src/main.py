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
from tensorflow.python import debug as tf_debug
import pickle as pk
def train(config):
    print("Building Model ... ")
    print("Setting Batch system ... ")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_batch_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle( 
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator,is_train=True)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(config.train_log_dir)
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        #sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        total_loss = 0.0
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        stack_grads = None
        cnt = 0
        for step in range(1, config.num_steps + 1):
            print('=================================================')
            enc_img, end_width, img_info, loss, grads, _ = sess.run( [model.enc_img, model.end_width, model.img_info, model.loss, model.grads, model.fin_output], feed_dict={handle: train_handle})# , model.stack_grads : 0} )
            if cnt == 0:
                stack_grads = grads
            else:
                for i, d in enumerate(stack_grads):
                    stack_grads[i] += grads[i]
            print( int(step % config.stack_batch) ,' | loss stack :',loss)
            #print('Model Info :', img_info)
            #print('Encoding Image :', np.shape(enc_img))
            #print('End Width :', end_width)
            #print(end_width)
            cnt += 1
            if step % config.stack_batch == 0:
                cnt = 0
                #for i, grads in enumerate(stack_grads):
                    #print('\n\n',i,'\n\n')
                    #print(np.shape(grads))
                    #print(grads)
                tt = int(step/config.stack_batch)
                print(int(step/config.stack_batch),': Stack loss Learning\n')
                result = list()
                for grads in stack_grads:
                    result.append( grads / config.stack_batch)
                tmp_dict = {handle:train_handle}
                for i, grad in enumerate(result):
                    tmp_dict[model.stack_grads[i]] = grad
                train_op, x, y = sess.run( [model.train_op, model.x_img, model.y_img], feed_dict=tmp_dict )
                print('===============================================')
                print('===============================================')
                x = x.astype(np.uint8)
                print(type(x))
                print(np.shape(x))
                print(x)
                print('-----------------------------------------------')
                y = y.astype(np.uint8)
                print(type(y))
                print(np.shape(y))
                print(y)
                print('===============================================')
                print('===============================================')
                img = Image.fromarray( x )
                img.save(open('img3/img_x_' + str(tt) + '.bmp','wb'))
                img_y = Image.fromarray( y )
                img_y.save(open('img3/img_y_' + str(tt) + '.bmp','wb'))
            '''
            if step % (config.stack_batch * config.check_point) == 0:
                sess.run(tf.assign(model.is_train, tf.constant(False,dtype=tf.bool)))
                img_list = list()
                for i in range(10):
                    images = sess.run( [model.final_output], feed_dict={handle:dev_handle} )
                    img_list.append(images)
            ''' 
                
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
            
            
            
