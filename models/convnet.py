from keras import models, optimizers
from keras.layers import  Input, Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout

def Convnet():
            # Model architecture
    model = models.Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(66, 200, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    #model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    return model

