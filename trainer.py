import glob
import os
import numpy as np
import cv2
import time

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.utils import plot_model

from data_loader import data_to_hdf5, data_from_hdf5, WORKING_DIR, DATA_DIR, HDF5_DIR, MODEL_DIR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def nvidia_model():

    model = Sequential()
    
    # ... Your Model

    return model


def train_model(model, X_train, y_train, compile_flag=True, is_rnn=False):
    
    # ... Your compile and fit functions and return trained model

    return model


def save_model(model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model.save(MODEL_DIR + '/model.h5')
    print("Model Saved!")


def main():
    mode = input(
        "\n\nPress [c] - Create new HDF5 from data or \nPress [t] - Train model using existing HDF5\nPress [ct] - Create HDF5 and Train model \n\tYour Selection: ")

    model = nvidia_model()

    model.summary()
    plot_model(model, show_shapes=True, to_file=WORKING_DIR + "/images/model.png")

    initial = True

    try:
        if 'c' in mode:
            data_to_hdf5()

        if 't' in mode:
            start_time = time.time()
            for batch, hfile in enumerate(glob.glob(HDF5_DIR + "/*.h5")):
                print("\n\nCurrent Batch: {}/{}".format(batch,len(glob.glob(HDF5_DIR + "/*.h5"))-1))

                X_train, y_train = data_from_hdf5(hfile)

                print("Batch Size:", np.array(X_train).shape, np.array(y_train).shape)

                if initial:
                    model = train_model(model, X_train, y_train)
                    initial = False
                else:
                    model = train_model(model, X_train, y_train, compile_flag=False)

            end_time = time.time()
            print("\n[info] Training time:", end_time-start_time)

            save_model(model)
            print("[info] Model saved in", MODEL_DIR)

    except KeyboardInterrupt:
        print("\n[info] Aborted by User")


if __name__ == "__main__":
    main()
