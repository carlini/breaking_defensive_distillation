import sys
import time
import tensorflow as tf
import numpy as np
from model import make_model, preprocess
from setup import *

def compute(dat,lab):
    BATCH_SIZE = 100
    img = tf.Variable(np.zeros((BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),dtype=np.float32))
    out = model(img)

    r = []
    for i in range(0,len(dat),BATCH_SIZE):
        data = img
        o = s.run(out, feed_dict={img: preprocess(dat[i:i+BATCH_SIZE])})
        r.extend(o)
    print(np.mean(np.abs(r)),np.std(np.abs(r)))
        

with tf.Session() as s:
    model = make_model(sys.argv[1],s)

    compute(test_data, test_labels)
