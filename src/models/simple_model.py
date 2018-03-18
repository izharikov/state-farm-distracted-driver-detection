from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def get_model(print_summary=True):
    # training s small convnet from scatch
    # convnet: a simple stack of 3 convolution layer with ReLU activation and followed by a max-pooling layers
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))  # 64 neurons
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # drop 50% of neurons

    # output layer: classify to 10 driver's states
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if (print_summary):
        model.summary()

    return model
