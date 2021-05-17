import glob
import time
import cv2
import os
import glob
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keract import get_activations, display_activations

# from data_loader import WORKING_DIR, MODEL_DIR
from preprocess import crop_scale
from data_loader import data_to_hdf5, data_from_hdf5, WORKING_DIR, HDF5_DIR, MODEL_DIR
from all_models import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# IMG_SIZE = (222, 780, 3)

def plot_history(all_history):
    plt.style.use('ggplot')

    train_loss = []
    val_loss = []
    for history in all_history:
        train_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])

    plt.plot(np.array(train_loss).flatten())
    plt.plot(np.array(val_loss).flatten())
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid('off')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('mse.png')
    plt.show()  

def visualize_features(model, img):
    # keract_inputs = img[:1]
    # keract_targets = target_test[:1]
    activations = get_activations(model, img)
    display_activations(activations, cmap="gray", save=False)


def train_model(model, X_train, y_train, compile_flag=True, is_rnn=False):
    learning_rate = 1e-4
    epochs = 10
    # loss_fn = keras.losses.MeanSquaredError()
    optimizer_fn = Adam(lr=learning_rate)

    if compile_flag:
        # Setting the parameters
        model.compile(optimizer=optimizer_fn,
                      loss='mse',
                      metrics=['mse', 'mae'])

    if is_rnn:
        callbacks_ = [callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1),
            callbacks.ModelCheckpoint(
                MODEL_DIR + '/model_' + time.strftime("%H%M%S") + '.h5',
                monitor='val_loss', save_best_only=True, verbose=1)]

        history_object = model.fit(
            X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=callbacks_)
    else:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history_object = model.fit(
            X_train, y_train, epochs=epochs, validation_split=0.2, shuffle=True, batch_size=128)

    # plot_history(history=history_object)
    print(history_object.history['loss'])
    print(history_object.history['val_loss'])

    return model, history_object


def save_model(model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model.save(MODEL_DIR + '/model.h5')
    print("Model Saved!")


def load_modelh5():
    return load_model(MODEL_DIR + '/model.h5')


def rnn_predictor(rnn_history=1):
    model = load_modelh5()

    img_list = []
    for img_name in glob.glob(WORKING_DIR + "/images/center*.jpg"):
        img = cv2.imread(img_name)
        img = crop_scale(img)
        img_list.append(img)

    # img = np.expand_dims(img, axis=0)  # img = img[np.newaxis, :, :]

    print(np.array(img_list).reshape(-1, rnn_history, 66, 200, 3).shape)
    img_list = np.array(img_list).reshape(-1, rnn_history, 66, 200, 3)
    print(model.predict(img_list))
    # steering = self.model.predict(img)


def predictor():
    img = cv2.imread(WORKING_DIR + "/images/center-2021-05-03T03:42:09.644815.jpg")
    img = crop_scale(img)
    img = np.expand_dims(img, axis=0)
    model = load_modelh5()
    print(model.predict(img))


def main():
    
    TRAINING_BATCH = 3000
    IS_RNN = False
    RNN_HISTORY = 1

    mode = input(
        "\n\nPress [c] - Create new HDF5 from data or \nPress [t] - Train model using existing HDF5\nPress [ct] - Create HDF5 and Train model \n\tYour Selection: ")

    model_select = input(
        "\nSelect your model:\n[0] NVIDIA\n[1] RNN\n[2] ResNet\n[3] VGG16\n\tYour selection: ")

    if '0' in model_select:
        model = nvidia_model()
    elif '1' in model_select:
        IS_RNN = True
        RNN_HISTORY = 5
        model = rnn_model(rnn_history=RNN_HISTORY)
    elif '2' in model_select:
        model = simple_resnet_model()
    elif '3' in model_select:
        model = mobilenet_model()

    initial = True
    X_train = []
    y_train = []
    overflow_X = []
    overflow_y = []
    loss_history_list = []

    try:
        if 'c' in mode:
            data_to_hdf5()

        if 't' in mode:
            model.summary()
            plot_model(model, show_shapes=True, to_file=WORKING_DIR + "/images/model.png")

            start_time = time.time()
            for batch, hfile in enumerate(glob.glob(HDF5_DIR + "/*.h5")):
                print("\n\nCurrent Batch: {}/{}".format(batch,
                      len(glob.glob(HDF5_DIR + "/*.h5"))-1))

                images, labels = data_from_hdf5(hfile)
                # cv2.imshow(" ", images[0])
                # cv2.waitKey()

                X_train = X_train + list(images)
                y_train = y_train + list(labels)
                

                if len(X_train) > TRAINING_BATCH:
                    loss_history = None

                    overflow_X = X_train[TRAINING_BATCH:]
                    overflow_y = y_train[TRAINING_BATCH:]

                    X_train = X_train[:TRAINING_BATCH]
                    y_train = y_train[:TRAINING_BATCH]

                    # X_train, y_train = equalize_data(data, labels)
                    if len(X_train) % RNN_HISTORY != 0:
                        X_train = X_train[: -(len(X_train) % RNN_HISTORY)]
                        y_train = y_train[: -(len(y_train) % RNN_HISTORY)]

                    print("Batch Size:", np.array(
                        X_train).shape, np.array(y_train).shape)

                    if initial:
                        if IS_RNN:
                            model, loss_history = train_model(model, np.array(X_train).reshape(-1, RNN_HISTORY, 66, 200, 3), np.array(y_train).reshape(-1, RNN_HISTORY, 1), is_rnn=True)
                        else:
                            model, loss_history = train_model(model, np.array(X_train), np.array(y_train))

                        initial = False
                    else:
                        if IS_RNN:
                            model, loss_history = train_model(model, np.array(X_train).reshape(-1, RNN_HISTORY, 66, 200, 3), np.array(y_train).reshape(-1, RNN_HISTORY, 1), compile_flag=False, is_rnn=True)
                        else:
                            model, loss_history = train_model(model, np.array(X_train), np.array(y_train), compile_flag=False)

                    X_train = overflow_X
                    y_train = overflow_y

                    print(loss_history.history['loss'])
                    print(loss_history.history['val_loss'])
                
                    loss_history_list.append(loss_history)

            end_time = time.time()
            print("\n[info] Training time:", end_time-start_time)
            
            save_model(model)
            print("[info] Model saved at", datetime.datetime.now())
        
        # predictor()
        model = load_modelh5()

        plot_history(loss_history_list)
        # visualize_features(model, img)

    except KeyboardInterrupt:
        print("\n[info] Aborted by User")


if __name__ == "__main__":
    main()
