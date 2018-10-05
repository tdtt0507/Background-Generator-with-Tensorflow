import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

activation = tf.nn.relu

def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]
    filters_out = c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']
    c['ksize'] = c['ksize_origin']
    with tf.variable_scope('A'):
        c['stride'] = c['block_stride']
        assert c['ksize'] == 3
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('B'):
        c['conv_filters_out'] = filters_out
        assert c['ksize'] == 3
        x = conv(x, c)
        x = bn(x, c)

    with tf.variable_scope('shortcut'):
        print('ShortCut')
        c['ksize'] = 1
        c['stride'] = c['block_stride']
        c['conv_filters_out'] = filters_out
        shortcut = conv(shortcut, c)
        shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

def deconv2d(input_, output_dim, ks=5, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)
def binary_crossentropy(t,o):
    eps=1e-8
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

def y_loss(x,y):
    loss = tf.abs( x - y )
    loss = tf.pow(loss, 2)
    loss = tf.reshape(loss, [-1] )
    loss = tf.reduce_mean(loss)
    return loss

# pixelwise vector normalization
def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm', reuse=tf.AUTO_REUSE):
        norm_x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
        return norm_x

# activation function
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# loss함수
def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))



# slim.conv2d -> filter를 shape에 맞춰서 다 만들고 w값 초기화하는 과정 생략
def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input_ = tf.cast(input_, tf.float32)
        print(input_)
        print(output_dim)
        print(ks)
        print(s)
        
        x = slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),  biases_initializer= None)
        return x

