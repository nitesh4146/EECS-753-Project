from keras.engine.input_layer import Input
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, GRU, LSTM
from keras.layers import Dropout, Activation, MaxPooling2D, TimeDistributed
from keras.layers import GlobalAveragePooling2D, add
from keras.optimizers import Adam, SGD, RMSprop
from keras import callbacks
from keras.utils import plot_model
from keras.applications import resnet50, vgg16, mobilenet


def nvidia_model():
    model = Sequential()    
    model.add(Lambda(lambda x: x / 255.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (3, 3), strides=(2, 2),
              padding='valid', activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),
              padding='valid', activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),
              padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1),
              padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1),
              padding='valid', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


def rnn_model(rnn_history=1):
    model = Sequential()

    model.add(TimeDistributed(Lambda(lambda x: x / 255.0),
              input_shape=(rnn_history, 66, 200, 3)))
    model.add(TimeDistributed(Conv2D(24, (5, 5), strides=(
        2, 2), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(36, (5, 5), strides=(
        2, 2), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(48, (5, 5), strides=(
        2, 2), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(
        1, 1), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(
        1, 1), padding='valid', activation='relu')))

    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, name="lstm_layer_rgb"))

    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(
        Dense(50, activation='relu'), name="first_dense_rgb"))
    model.add(TimeDistributed(Dense(10, activation='relu')))
    model.add(TimeDistributed(Dense(1)))

    return model


def vgg16_model():
    model = Sequential()
    input_layer = Input(shape=(66, 200, 3))

    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_layer)

    model.add(vgg_model)
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(126, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    for layer in vgg_model.layers:
        layer.trainable = False

    return model

def resnet50_model():
    input_layer = Input(shape=(66, 200, 3))
    resnet_model = resnet50.ResNet50(include_top=False, weights="imagenet", input_tensor=input_layer)

    for layer in resnet_model.layers:
        layer.trainable = False

    for i, layer in enumerate(resnet_model.layers):
        print(i, layer.name, "-", layer.trainable)

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=(66, 200, 3)))
    model.add(resnet_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(126, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def simple_resnet_model():

    input_1 = Input(shape=(66, 200, 3))
    lambda_1 = Lambda(lambda x: x / 255.0)(input_1)
    # conv1_pad = ZeroPadding2D()(input_1)
    conv1 = Conv2D(32, (5,5), padding='same')(input_1)
    bn_conv1 = BatchNormalization()(conv1)
    activation_1 = Activation('relu')(bn_conv1)
    # pool1_pad = ZeroPadding2D()(activation_1)
    max_pool2d_1 = MaxPooling2D()(activation_1)

    res2a_branch1 = Conv2D(64, (3,3), padding='same', strides=(2,2))(max_pool2d_1)
    bn2a_branch1 = BatchNormalization()(res2a_branch1)

    res2a_branch2a = Conv2D(32, (5,5), padding='same', strides=(2,2))(max_pool2d_1)
    bn2a_branch2a = BatchNormalization()(res2a_branch2a)
    activation_2 = Activation('relu')(bn2a_branch2a)
    res2a_branch2b = Conv2D(32, (5,5), padding='same')(activation_2)
    bn2a_branch2b = BatchNormalization()(res2a_branch2b)
    activation_3 = Activation('relu')(bn2a_branch2b)
    res2a_branch2c = Conv2D(64, (3,3), padding='same')(activation_3)
    bn2a_branch2c = BatchNormalization()(res2a_branch2c)

    add_1 = add([bn2a_branch2c, bn2a_branch1])
    activation_4 = Activation('relu')(add_1)

    res2b_branch2a = Conv2D(64, (5,5), padding='same')(activation_4)
    bn2b_branch2a = BatchNormalization()(res2b_branch2a)
    activation_5 = Activation('relu')(bn2b_branch2a)
    res2b_branch2b = Conv2D(64, (5,5), padding='same')(activation_5)
    bn2b_branch2b = BatchNormalization()(res2b_branch2b)
    activation_6 = Activation('relu')(bn2b_branch2b)
    res2b_branch2c = Conv2D(64, (3,3), padding='same')(activation_6)
    bn2b_branch2c = BatchNormalization()(res2b_branch2c)

    add_2 = add([bn2b_branch2c, activation_4])
    activation_7 = Activation('relu')(add_2)

    res2c_branch1 = Conv2D(64, (3,3), padding='same', strides=(2,2))(activation_7)
    bn2c_branch1 = BatchNormalization()(res2c_branch1)

    res2c_branch2a = Conv2D(64, (5,5), padding='same', strides=(2,2))(activation_7)
    bn2c_branch2a = BatchNormalization()(res2c_branch2a)
    activation_8 = Activation('relu')(bn2c_branch2a)
    res2c_branch2b = Conv2D(64, (5,5), padding='same')(activation_8)
    bn2c_branch2b = BatchNormalization()(res2c_branch2b)
    activation_9 = Activation('relu')(bn2c_branch2b)
    res2c_branch2c = Conv2D(64, (3,3), padding='same')(activation_9)
    bn2c_branch2c = BatchNormalization()(res2c_branch2c)

    add_3 = add([bn2c_branch2c, bn2c_branch1])
    activation_10 = Activation('relu')(add_3)

    global_pool1 = GlobalAveragePooling2D()(activation_7)
    dense1 = Dense(1024, activation='relu')(global_pool1)
    drop1 = Dropout(0.3)(dense1)
    # dense12 = Dense(512, activation='relu')(dense1)
    dense2 = Dense(256, activation='relu')(dense1)
    drop2 = Dropout(0.3)(dense2)
    dense3 = Dense(50, activation='relu')(drop2)
    # drop3 = Dropout(0.3)(dense3)
    dense4 = Dense(10, activation='relu')(dense3)
    output_layer = Dense(1)(dense4)

    model = Model(inputs=input_1, outputs=output_layer)
    # plot_model(model, show_shapes=True, to_file=WORKING_DIR + "/images/my_resnet.png")

    return model


def mobilenet_model():
    base_mobilenet_model = mobilenet.MobileNet(input_shape =  (66, 200, 3), 
                                 include_top = False, 
                                 weights = None)
    model = Sequential()
    model.add(BatchNormalization(input_shape = (66, 200, 3)))
    model.add(base_mobilenet_model)
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model
    
