import  tensorflow as tf
import  numpy as np
import  sys
import glob
import  os
import  random
import math
import collections
import time
import shutil
# import matplotlib.pyplot as plt
########################################################################slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
slim = tf.contrib.slim







def conv(batch_input, out_channels, stride=1):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [5, 5, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv

def conv_s(batch_input, out_channels, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="VALID"):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [ksize_x, ksize_y, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stridex, stridey, 1], padding)
        return conv

def dense(input, n_output):
    with tf.variable_scope("dense"):
        weights = tf.get_variable("weights",[input.get_shape()[1], n_output], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        biases = tf.get_variable("biases",shape=n_output, dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.01))
    return tf.matmul(input, weights)+biases

def dense_s(input, n_output):
    with tf.variable_scope("dense"):
        weights = tf.get_variable("weights",[input.get_shape()[1], n_output], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        biases = tf.get_variable("biases",shape=n_output, dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.01))
    return tf.matmul(input, weights)+biases

def maxpool_s(batch_input,  ksize_x=2,ksize_y=2,stridex=2,stridey=2, padding="SAME"):
    with tf.variable_scope("maxpool"):
        x = tf.nn.max_pool(batch_input, ksize=[1, ksize_x, ksize_y, 1], strides=[1, stridex, stridey, 1], padding='SAME')
        return x

def _normalize( x):
    w = tf.shape(x)[1]
    _max = tf.reduce_max(x, axis=[1, 2, 3])
    _max = tf.reshape(_max, [-1, 1, 1, 1])
    _max = tf.tile(_max, [1, w, w, 1])
    _min = tf.reduce_min(x, axis=[1, 2, 3])
    _min = tf.reshape(_min, [-1, 1, 1, 1])
    _min = tf.tile(_min, [1, w, w, 1])
    x = (x - _min)/(_max - _min + 1e-10)
    return x



def batchnorm(input, isT):

    normalized = tf.contrib.layers.batch_norm(input, updates_collections=None, decay=0.9, center=True, scale=True, is_training=isT)
    return normalized




def create_tower_2mp_5c(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')


    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    return x


def create_tower_2mp_5c_hg(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    return x

def create_tower_2mp_5c_norelu(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        # x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        # x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x = tf.concat( [x1, x2],3)
        x=  tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        # x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        # x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x = tf.concat( [x1, x2],3)

        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)



    x=tf.squeeze(x)


    return x



def create_tower_2mp_5c_hg_f1024_f256(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)



    x=tf.squeeze(x)

    # with tf.variable_scope("layer_6"):
    #     x= dense(x,256)


    return x


def create_tower_2mp_5c_hg_norelu(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    return x



def create_tower_2mp_5c_2fc_hg_norelu_256(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    with tf.variable_scope("layer_6"):
        x=dense_s(x, 256)
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)


    return x

def create_tower_2mp_5c_2fc_hg_norelu_64(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    with tf.variable_scope("layer_6"):
        x=dense_s(x, 64)
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)


    return x



def create_tower_2mp_5c_2fc_hg_norelu_1024(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x1=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    with tf.variable_scope("layer_6"):
        x=dense_s(x, 1024)
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)


    return x








def create_tower_2mp_5c_2fc_norelu_1024(inputs,isTraining,keep_prob=0.5):


    with tf.variable_scope("layer_1"):
        x=conv_s(inputs, 64, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        # x = tf.nn.conv2d(inputs, [3,3,3,64],[1, 1, 1, 1] , padding='VALID')
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_1"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')

    with tf.variable_scope("layer_2"):
        x=conv_s(x, 128, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_2"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x = tf.concat( [x1, x2],3)

    with tf.variable_scope("layer_3"):
        x=conv_s(x, 256, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool_3"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')



    with tf.variable_scope("layer_41"):
        x=conv_s(x, 512, ksize_x=3,ksize_y=3, stridex=1,stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)

    with tf.variable_scope("layer_42"):
        x = conv_s(x, 512, ksize_x=3, ksize_y=3, stridex=1, stridey=1, padding="SAME")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)


    with tf.variable_scope("pool_4"):
        x=tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='VALID')
        x=  tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x2 = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
        # x = tf.concat( [x1, x2],3)
    with tf.variable_scope("layer_5f"):
        x=conv_s(x, 1024, ksize_x=1,ksize_y=10, stridex=1,stridey=1, padding="VALID")
        x = batchnorm(x, isTraining)
        x = tf.nn.relu(x)
    with tf.variable_scope("layer_5t"):
        x = tf.reduce_mean(x, axis=1, keep_dims=True)

    x=tf.squeeze(x)


    with tf.variable_scope("layer_6"):
        x=dense_s(x, 1024)
        # x = batchnorm(x, isTraining)
        # x = tf.nn.relu(x)


    return x