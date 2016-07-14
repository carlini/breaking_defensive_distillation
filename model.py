## model.py -- an 8 layer neural network with 726k paramaters
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


import tensorflow as tf
import pickle
import numpy as np

def make_model(restore, is_train=tf.constant(False), NUM_CHANNELS=1, IMAGE_SIZE=28, NUM_LABELS=10):
    if restore == None:
        conv1_weights = tf.Variable(
            tf.truncated_normal([3, 3, NUM_CHANNELS, 32], stddev=0.1))
        conv1_beta = tf.Variable(tf.zeros(32))
        
        conv2_weights = tf.Variable(
            tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
        conv2_beta = tf.Variable(tf.zeros(32))
        
        conv3_weights = tf.Variable(
            tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
        conv3_beta = tf.Variable(tf.zeros(64))
        
        conv4_weights = tf.Variable(
            tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
        conv4_beta = tf.Variable(tf.zeros(64))
        
        fc1_weights = tf.Variable(
            tf.truncated_normal([7 * 7 * 64, 200], stddev=0.1))
        fc1_beta = tf.Variable(tf.zeros(200))
        
        fc2_weights = tf.Variable(
            tf.truncated_normal([200, 200], stddev=0.1))
        fc2_beta = tf.Variable(tf.zeros(200))
        
        fc3_weights = tf.Variable(
            tf.truncated_normal([200, NUM_LABELS], stddev=0.1))
        fc3_beta = tf.Variable(tf.zeros(NUM_LABELS))
        
    else:
        conv1_weights, conv1_beta, conv2_weights, conv2_beta, conv3_weights, conv3_beta, conv4_weights, conv4_beta, fc1_weights, fc1_beta, fc2_weights, fc2_beta, fc3_weights, fc3_beta  = [tf.constant(np.array(x,dtype=np.float32)) for x in pickle.load(open(restore,"rb"))]

    # This model and the paramaters are the same as those defined in
    # Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    # Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami
    # Available at
    # https://arxiv.org/pdf/1511.04508.pdf
    
    def model(data, train=False):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv+conv1_beta)
        
        conv = tf.nn.conv2d(relu, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv+conv2_beta)
        
        pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv+conv3_beta)
        
        conv = tf.nn.conv2d(relu, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv+conv4_beta)
        
        pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool,
                             [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights)+fc1_beta)
        
        hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights)+fc2_beta)
        
        if train:
            hidden = tf.nn.dropout(hidden, .5)

        return tf.matmul(hidden, fc3_weights) + fc3_beta

    def saver(s, nn):
        def deeplist(x):
            try:
                x[0]
                return list(map(deeplist,x))
            except:
                return x
        
        dd = [deeplist(s.run(x)) for x in [conv1_weights, conv1_beta, conv2_weights, conv2_beta, conv3_weights, conv3_beta, conv4_weights, conv4_beta, fc1_weights, fc1_beta, fc2_weights, fc2_beta, fc3_weights, fc3_beta]]
        pickle.dump(dd,open(nn,"wb"),pickle.HIGHEST_PROTOCOL)

    if restore:
        return model
    else:
        return model, saver

def preprocess(x):
  return x
