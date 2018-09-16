import tensorflow as tf
import os
import argparse as _argparse
from PIL import Image
import numpy as np
import collections
import copy
import random
from tqdm import tqdm

def list_check(src, dst):
    if len(src) != len(dst):
        return False
    for idx, s in enumerate(src):
        if s != dst[idx]:
            return False
    return True
        
def filtered_v01(image):
    return None

def detect_object(image):
    # input : checked image
    pix = np.array(image)
    print('np.shape()')
    ## [ 세로 ] [ 가로 ] [ 채널 ]
    print(np.shape(pix))
    # print(len(pix))
    # print(len(pix[0])) # [height][width][channel]
    # print(len(pix[0][0]))
    width  = [ [0, 0], [0, 0] ]   # width's  [start point[x, y] , end point[x,y]]
    height = [ [0, 0], [0, 0] ]   # height's [start point[x, y] , end point[x, y]]
    
    # Calculating Width 
    for i in range( len(pix) ): # 
        start = [0, 0]
        keep = 0
        out = 0
        for j in range( len(pix[0]) ):
            if list_check(pix[i][j],[255,0,0]) and list_check(start,[0,0]):#(start == [0, 0]) && 
                start = [i, j]
                keep = 1
            if keep > 0 and list_check(pix[i][j],[255, 0, 0]):
                end = [i, j]
                keep += 1
            if keep > 50 and not list_check(pix[i][j], [255, 0, 0]):         
                width[0] = start
                width[1] = end
                out = 1
                break
        if out == 1:
            break
            
    for j in range( len(pix[0]) ):
        start = [0,0]
        keep = 0
        out = 0
        for i in range( len(pix) ):
            if list_check(start,[0, 0]) and list_check(pix[i][j], [255,0,0]):
                start = [i, j]
                keep = 1
            if keep > 0 and list_check(pix[i][j], [255, 0, 0]):
                end = [i, j]
                keep += 1
            if keep > 50 and not list_check(pix[i][j], [255, 0, 0]):         
                height[0] = start
                height[1] = end
                out = 1
                break
        if out == 1:
            break
    # { loc : (y, x), size : (width, height)}
    return width[0] + list((width[1][1] - width[0][1], height[1][0] - height[0][0]))    

def get_plus_info(obj_info, img_info):
    plus_info = {}

    plus_info['left']  = random.randrange(0, obj_info['x'])
    plus_info['right'] = random.randrange(0, img_info['w'] - (obj_info['x'] + obj_info['w']))
    plus_info['down']  = random.randrange(0, img_info['h'] - (obj_info['y'] + obj_info['h']))
    plus_info['up']    = random.randrange(0, obj_info['y'])

    return plus_info

def setting_etc(image_pix, plus_info, t_obj_info):
    output_pix = copy.deepcopy(image_pix)
    output_pix = output_pix[t_obj_info['y'] - plus_info['up'] : t_obj_info['y'] + t_obj_info['h'] + plus_info['down'], :]
    output_pix = output_pix[:, t_obj_info['x'] - plus_info['left'] : t_obj_info['x'] + t_obj_info['w'] + plus_info['right']]
    t_obj_info['x'] = plus_info['left']    
    t_obj_info['y'] = plus_info['up']

    return output_pix, t_obj_info

def augment_image(dataset, image, obj_info, num, cnt):

    possible = {'up':0,'down':0,'left':0,'right':0} # [상, 하, 좌, 우]
    image_pix = np.array(image)
    img_info = {'x':0, 'y':0, 'w':len(image_pix[0]), 'h':len(image_pix)}
    
    min_gen = 150
    # 최소 여백 부분을 100, 최소 생성 부분 50, 따라서 150으로 설정
    possible['up'] = 1 if obj_info['y'] > min_gen else 0
    possible['down'] = 1 if obj_info['y'] + obj_info['h'] + min_gen < img_info['h'] else 0
    possible['left'] = 1 if obj_info['x'] > min_gen else 0
    possible['right'] = 1 if obj_info['x'] + obj_info['w'] + min_gen < img_info['w'] else 0

    x_data_s = list()
    y_data_s = list()

    data_bundle = list()
    
    if possible['up']: # 위로만 생성 데이터
        for i in range(num):
            input_pix = None
            output_pix = None
            t_obj_info = copy.deepcopy(obj_info)
            plus_info = get_plus_info(t_obj_info, img_info)
            plus_info['up']    = t_obj_info['y']

            output_pix, t_obj_info = setting_etc(image_pix, plus_info, t_obj_info)
            input_pix = copy.deepcopy(output_pix)
            erase_area = random.randrange(50, t_obj_info['y'] - 100)
            input_pix[: erase_area, : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info, 'zero':'up', 'id':cnt})
            cnt += 1
    
    if possible['down']: # 아래로만 생성 데이터
        for i in range(num):
            input_pix = None
            output_pix = None
            t_obj_info = copy.deepcopy(obj_info)
            plus_info = get_plus_info(t_obj_info, img_info)
            plus_info['down'] = img_info['h'] - (t_obj_info['y'] + t_obj_info['h'])

            output_pix, t_obj_info = setting_etc(image_pix, plus_info, t_obj_info)
            input_pix = copy.deepcopy(output_pix)
            erase_area = random.randrange(50, len(output_pix) - 100 - t_obj_info['h'] - t_obj_info['y'])
            input_pix[ len(output_pix) - erase_area: , : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info, 'zero':'down', 'id':cnt})
            cnt += 1
    
    if possible['left']:# 왼으로만 생성 데이터
        for i in range(num):
            input_pix  = None
            output_pix = None
            t_obj_info = copy.deepcopy(obj_info)
            plus_info = get_plus_info(t_obj_info, img_info)
            plus_info['left'] = t_obj_info['x']

            output_pix, t_obj_info = setting_etc(image_pix, plus_info, t_obj_info)
            input_pix  = copy.deepcopy(output_pix)
            erase_area = random.randrange(50, t_obj_info['x'] - 100)
            input_pix[:, :erase_area ] = 0

            im_input  = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info, 'zero': 'left', 'id':cnt})
            cnt += 1
    
    if possible['right']:
        for i in range(num):
            input_pix = None 
            output_pix = None
            t_obj_info = copy.deepcopy(obj_info)
            plus_info = get_plus_info(t_obj_info, img_info)
            plus_info['right'] = img_info['w'] - (t_obj_info['x'] + t_obj_info['w'])

            output_pix, t_obj_info = setting_etc(image_pix, plus_info, t_obj_info)
            input_pix = copy.deepcopy(output_pix)
            erase_area = random.randrange(50, len(output_pix[0]) - (t_obj_info['x'] + t_obj_info['w'] + 100) )
            input_pix[:, len(output_pix[0]) - erase_area : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info, 'zero':'right', 'id':cnt})
            cnt += 1
    return data_bundle, cnt

def make_tf_record(dataset, kind):
    cnt = 0
    data_list = os.listdir(dataset +'/original')
    tmp = list()
    input_tf = list()
    for data in data_list:
        if data.split("_")[1] != 'c':
            tmp.append(data)
    data_tuple = list()

    for data in tmp:
        c_data = data.split('_')
        c_data = c_data[0] +'_c_' + c_data[1]
        data_tuple.append( (data, c_data) )

    for data in data_tuple:
        print(data)
        origin = Image.open(os.path.join(dataset + '/original', data[0])) # Original Image
        chked  = Image.open(os.path.join(dataset + '/original', data[1])) # object detected Image        
        obj_info = detect_object(chked) # obj_info = [y, x, width, height]
        obj_info_ = {'x':obj_info[1],'y':obj_info[0],'w':obj_info[2],'h':obj_info[3]}
        obj_info = obj_info_
        ## 기본적인 생성방식, N 개의 image 를 random 방식으로 잘라서 data 를 augment
        N = 10 # 데이터 증강 개수
        tmp_tf, cnt = augment_image(dataset + '/augment', origin, obj_info, N, cnt)             
        input_tf += tmp_tf
        N = 10
    return input_tf

def produce_tfrecord(config, examples):
    print('Generating train.tfrecord')
    writer = tf.python_io.TFRecordWriter(config.train_record_file)    
    for example in examples:
        print(example)
        img_x = Image.open(example['X'])
        img_y = Image.open(example['Y'])
        np_x = np.array(img_x, dtype=np.int32)
        np_y = np.array(img_y, dtype=np.int32)
        obj_info = example['obj_info']
        obj_info_ = np.array( [obj_info['x'], obj_info['y'], obj_info['w'], obj_info['h']], dtype=np.int32)
        
        if example['zero'] == 'up':
            np_x = np_x[:obj_info['y'],:]
            np_y = np_y[:obj_info['y'],:]
            np_x = np.rot90(np_x, 3)
            np_y = np.rot90(np_y, 3)
        elif example['zero'] == 'down':
            np_x = np_x[obj_info['y'] + obj_info['h']:,:]
            np_y = np_y[obj_info['y'] + obj_info['h']:,:]
            np_x = np.rot90(np_x, 1)
            np_y = np.rot90(np_y, 1)
        elif example['zero'] == 'left':
            np_x = np_x[:,:obj_info['x']]
            np_y = np_y[:,:obj_info['x']]
            np_x = np.rot90(np_x, 2)
            np_y = np.rot90(np_y, 2)
        elif example['zero'] == 'right':
            np_x = np_x[:, obj_info['x'] + obj_info['w']:]
            np_y = np_y[:, obj_info['x'] + obj_info['w']:]
        
        end_width = 0
        for idx, val in enumerate(np_x[0]):
            if list_check(val,[0,0,0]):
                end_width = idx
                break
        img_info = np.array( [0,0, len(np_x[0]), len(np_x) ], dtype=np.int32)
        
        features = tf.train.Features(feature={
            # 실제 이미지 Location
            "id" : tf.train.Feature(int64_list=tf.train.Int64List(value=[example['id']])),
            # 생성할 부분만 잘린 Image Pixel
            "c_image" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_x.tostring()])),
            "r_image" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_y.tostring()])),
            "img_info" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_info.tostring()])),
            "end_width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[end_width]))
            })
        record = tf.train.Example(features=features)
        writer.write(record.SerializeToString())
    writer.close()
    
def prepro(config):

    print('Preprocessing ... ')
    train_dir = config.train_dir
    dev_dir = config.dev_dir
    test_dir = config.test_dir
    
    # prepro 모듈 동작 시나리오

    train_input_tf = make_tf_record(train_dir,"Training data")
    produce_tfrecord(config, train_input_tf)
    