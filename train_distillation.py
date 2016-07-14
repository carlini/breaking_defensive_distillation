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
import train_baseline
from model import make_model
from setup import *

def train(train_data, train_labels, teacher_name, file_name,
          NUM_EPOCHS=50, BATCH_SIZE=128, TRAIN_TEMP=1):
    # Step 1: train the teacher model
    train_baseline.train(train_data, train_labels, teacher_name,
                         NUM_EPOCHS, BATCH_SIZE, TRAIN_TEMP)


    # Step 2: evaluate the model on the training data at the training temperature
    soft_labels = np.zeros(train_labels.shape)
    with tf.Session() as s:
        model = make_model(teacher_name)
        xs = tf.placeholder(tf.float32,
                            shape=(100, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        predictions = tf.nn.softmax(model(xs)/TRAIN_TEMP)

        for i in range(0,len(train_data),100):
            predicted = predictions.eval({xs: train_data[i:i+100]})
            soft_labels[i:i+100] = predicted

    # Step 3: train the distilled model on the new training data
    train_baseline.train(train_data, soft_labels, file_name,
                         NUM_EPOCHS, BATCH_SIZE, TRAIN_TEMP)
    
        
teacher_name = "models/teacher"
file_name = "models/distilled"
train(train_data, train_labels, teacher_name, file_name,
      NUM_EPOCHS=50, BATCH_SIZE=128, TRAIN_TEMP=100)
