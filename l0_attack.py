## l0_attack.py -- Papernot et al.'s l0 attack to find adversarial examples
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

import random
import sys
import time
import tensorflow as tf
import numpy as np

from setup import *
from model import make_model

def modified_papernot_attack(imgs, labs, TEMPERATURE):
    BATCH_SIZE = 10

    delta = tf.Variable(tf.zeros((BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)))
    img = tf.placeholder(tf.float32, (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
    lab = tf.placeholder(tf.float32, (BATCH_SIZE,10))
    
    out = tf.nn.softmax(model(img+delta)/TEMPERATURE)

    target_probability = tf.reduce_sum(out*lab,0)
    other_probability = tf.reduce_sum(out*(1-lab),0)

    grads_target = tf.gradients(target_probability, [delta])[0]
    grads_other = tf.gradients(other_probability, [delta])[0]

    s.run(tf.global_variables_initializer())

    total = []
    costs = []
    for offset in range(0,len(imgs),BATCH_SIZE):
        obatch_imgs = imgs[offset:offset+BATCH_SIZE]
        batch_labs = labs[offset:offset+BATCH_SIZE]

        batch_imgs = np.copy(obatch_imgs)
        used = np.zeros((BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE))

        # 1. randomly sample a target which is not the current one
        targets = batch_labs
        while np.sum(targets*batch_labs) != 0:
            targets = np.array([np.identity(10)[random.randint(0,9)] for _ in range(BATCH_SIZE)])

        # 2. Try changing pixels up to 112 times
        for _ in range(113):

            # 3. Find which ones we've already succeeded on.
            the_outs = s.run(out, feed_dict={img: batch_imgs})
            success = np.argmax(the_outs,axis=1) == np.argmax(targets,axis=1)

            if np.sum(success) == BATCH_SIZE:
                # abort early if we are done
                break
            
            # 4. Compute the gradients required (alpha and beta)
            dir_targ, dir_other = s.run([grads_target, grads_other], 
                                        feed_dict={img: batch_imgs, lab:targets})


            for e in range(BATCH_SIZE):
                if not success[e]:
                    # 5. Pick the next most important pixel we can change
                    
                    directions = (-dir_other+dir_targ)
                    while True:
                        dirs = np.sum(np.abs(directions),axis=3) * (np.sum(dir_other,axis=3) > 0) * (np.sum(dir_targ,axis=3) < 0) * (1-used[e])
                        
                        highest = np.argmax(dirs[e,:,:])
                        x,y = highest%IMAGE_SIZE, highest//IMAGE_SIZE

                        curval = batch_imgs[e,y,x,:]
                        
                        change = np.sign(directions[e,y,x])
                        
                        if np.all(change == 0):
                            break
                            
                        if abs(curval+change) < 1.499:
                            # 6. Actually change it by the right direction

                            used[e,y,x] += 1
                            batch_imgs[e,y,x,:] += change
                            batch_imgs = np.clip(batch_imgs, -.5, .5)
                            break
                        else:
                            directions[e,y,x] = 0

        # Recompute the success probability
        the_outs = s.run(out, feed_dict={img: batch_imgs})
        success = (np.argmax(the_outs,axis=1) == np.argmax(targets,axis=1))
        
        # Count the number of pixels we had to change
        different = (batch_imgs!=obatch_imgs).reshape((BATCH_SIZE,IMAGE_SIZE**2,NUM_CHANNELS))
        different = np.any(different,axis=2)
        
        # And success requires we change fewer than 112 pixels.
        success &= np.sum(different,axis=1) < 112

        #for e in range(BATCH_SIZE):
        #    print('out',np.argmax(batch_labs[e]), np.argmax(targets[e]), np.sum(different[e]), success[e])

        costs.extend(np.sum(different,axis=1))
        total.extend(success)
        
        print(np.mean(costs),np.mean(total))
    

if __name__ == "__main__":
    with tf.Session() as s:
        model = make_model(sys.argv[1])
        print("Number of pixels changed / Probability of Attack Success")
        print(modified_papernot_attack(test_data[:10000], test_labels[:10000], 100))

