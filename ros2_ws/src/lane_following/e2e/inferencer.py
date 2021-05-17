import numpy as np
import cv2
import os
import csv
import glob
import progressbar

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import time
from preprocess import crop_scale

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/inference/"

MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + "/images/models/"
MODEL_FILE = "model_m1.h5"
IMG_PATH = os.path.dirname(os.path.realpath(__file__)) + "/images/img/"

model_name = "rnn"
model = load_model(MODEL_PATH + model_name + "/" + MODEL_FILE)


# @profile
def inference():
    img_set = []

    with progressbar.ProgressBar(max_value=len(glob.glob(IMG_PATH + "*.jpg"))) as bar:
        for i, img_path in enumerate(glob.glob(IMG_PATH + "*.jpg")):
            # print(img_path)
            inference_time = 0.
            img = cv2.imread(img_path)


            img = crop_scale(img)

            # ## CNN Specific Block
            # img = np.expand_dims(img, axis=0)  # img = img[np.newaxis, :, :]
            # t0 = time.time()
            # steering = model.predict(img)
            # t1 = time.time()
            # inference_time = t1 - t0
            # ## CNN Specific Block

            ## RNN Specific Block
            img_set.append(img)
            steering = 0.

            if len(img_set) == 5:
                img_set_ = np.expand_dims(np.array(img_set), axis=0)  
                # print(img_set_.shape)
                
                t0 = time.time()
                steering = model.predict(img_set_)[0][-1]
                t1 = time.time()
                inference_time = t1 - t0

                img_set.pop(0)
            ## RNN Specific Block

            

            with open(DATA_DIR + "inference_time_" + model_name + ".csv", 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([inference_time])

            bar.update(i)

if __name__ == '__main__':
    inference()