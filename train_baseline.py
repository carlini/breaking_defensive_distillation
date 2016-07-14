## train_baseline.py -- train a standard MNIST classifier at 99.5% accuracy
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

import numpy as np
import tensorflow as tf
from model import make_model
from setup import *

def train(train_data, train_labels, file_name, NUM_EPOCHS=50, BATCH_SIZE=128, TRAIN_TEMP=1):
    train_size = train_labels.shape[0]
    
    train_xs = tf.placeholder(tf.float32,
                              shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_ys = tf.placeholder(tf.float32,
                              shape=(BATCH_SIZE, NUM_LABELS))
    
    model, saver = make_model(None, NUM_CHANNELS=NUM_CHANNELS, IMAGE_SIZE=IMAGE_SIZE)
    
    logits = model(train_xs, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits/TRAIN_TEMP, train_ys))
    
    lr = tf.placeholder(tf.float32, [])
    momentum = tf.placeholder(tf.float32, [])
    
    optimizer = tf.train.MomentumOptimizer(lr, momentum).minimize(loss)
    
    check_xs = tf.placeholder(tf.float32,
                              shape=(100, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    check_prediction = tf.nn.softmax(model(check_xs))


    def check(data, labels):
        err = []
        for i in range(0,len(data),100):
            predicted = check_prediction.eval({check_xs: data[i:i+100]})
            valids = np.argmax(predicted, 1) == np.argmax(labels[i:i+100], 1)
            err.extend(valids)
        return 1-np.mean(err)
        
    
    with tf.Session() as s:
        tf.initialize_all_variables().run()
        
        for step in range(NUM_EPOCHS * train_size // BATCH_SIZE):
            epoch = (float(step) * BATCH_SIZE / train_size)
            
            offset = np.random.random_integers(0, train_size-1, BATCH_SIZE)
            batch_data = train_data[offset, :, :, :]
            batch_labels = train_labels[offset, :]
            
            feed_dict = {train_xs: batch_data,
                         train_ys: batch_labels,
                         lr: (0.5**int(epoch/10))*0.01,
                         momentum: (0.5**int(epoch/10))*0.9}
            
            s.run(optimizer, feed_dict=feed_dict)
            
            if step % 100 == 0:
                print("Step %d/%d"%(step, NUM_EPOCHS * train_size // BATCH_SIZE))

                print("Validation error: ", check(validation_data, validation_labels))

        print("\nTest error: ", check(test_data, test_labels))
        saver(s, file_name)
    
if __name__ == "__main__":
    train(train_data, train_labels, "models/baseline", NUM_EPOCHS=20)
