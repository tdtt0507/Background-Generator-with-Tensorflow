import os
import numpy as np
from PIL import Image
import collections
import copy
import random

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
    print(obj_info)
    min_gen = 150
    # 최소 여백 부분을 150, 최소 생성 부분 50, 따라서 200으로 설정
    possible['up'] = 1 if obj_info['y'] > min_gen else 0
    possible['down'] = 1 if obj_info['y'] + obj_info['h'] + min_gen < img_info['h'] else 0
    possible['left'] = 1 if obj_info['x'] > min_gen else 0
    possible['right'] = 1 if obj_info['x'] + obj_info['w'] + min_gen < img_info['w'] else 0

    print('============ produce_tfrecord ============')
    print('Object Image \n X : ' +  str(obj_info['x']) + '\n Y : ' + str(obj_info['y']) + '\n Width : ' + str(obj_info['w']) + '\n Height : ' + str(obj_info['h']) )
    print('Input  Image \n X : 0 \n Y : 0 \n Width : ' + str(img_info['w']) + ' \n Height : ' + str(img_info['h']) + '\n' )
    print('==========================================')
    print(possible)
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
            erase_area = random.randrange(0, t_obj_info['y'] - 100)
            input_pix[: erase_area, : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info})
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
            erase_area = random.randrange(0, len(output_pix) - 100 - t_obj_info['h'] - t_obj_info['y'])
            input_pix[ len(output_pix) - erase_area: , : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info})
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
            erase_area = random.randrange(0, t_obj_info['x'] - 100)
            input_pix[:, :erase_area ] = 0

            im_input  = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info})
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
            erase_area = random.randrange(0, len(output_pix[0]) - (t_obj_info['x'] + t_obj_info['w'] + 100) )
            print('-------output_pix[0]-------')
            print(np.shape(input_pix))
            print(len(output_pix[0]))
            print('-------erase_area-----------')
            print(erase_area)
            print('---------------------')
            input_pix[:, len(output_pix[0]) - erase_area : ] = 0
            im_input = Image.fromarray(input_pix)
            im_output = Image.fromarray(output_pix)
            x_file = dataset + '/image_X_' + str(cnt) + '.bmp'
            y_file = dataset + '/image_Y_' + str(cnt) + '.bmp'
            im_input.save( x_file )
            im_output.save( y_file )
            data_bundle.append({'X': x_file, 'Y':y_file, 'obj_info' : t_obj_info})
            cnt += 1

    return data_bundle, cnt

def produce_tf_record():
    return None

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
        tmp_tf, cnt = augment_image(dataset + '/augment', origin, obj_info, N, cnt) # 원래 이미지, 오브젝 정보, 데이터 증강 개수
            
        input_tf += tmp_tf

        ## 추가구현 생성방식, N 개의 image 를 필터를 거쳐 변형시킨 이미지를 잘라서 data augment
        #origin = filtered_v01(origin)
        N = 10
    produce_tfrecord(input_tf)
        
    print(kind + ' --> tfrecord 생성 완료')

def prepro(config):
    print('=== prepro ===')
    train_dir = config.train_dir
    dev_dir = config.dev_dir
    test_dir = config.test_dir
    processed_dir = config.processed_dir
    
    # prepro 모듈 동작 시나리오
    make_tf_record(train_dir,"Training data")
    make_tf_record(dev_dir,"Dev data")
    make_tf_record(test_dir,"Test data")
    ## 
    '''
    에러가 있어서 향후 수정 예정, 나중에 이부분으로 작업할 예정.
    # Raw dataset directory
    train_file_dir = config.train_dir
    dev_file_dir = config.dev_dir
    test_file_dir = config.test_dir
    
    # preprocessed dataset directory
    train_record_file = config.train_record_file
    dev_record_file = config.dev_record_file
    test_record_file  = config.test_record_file
    '''