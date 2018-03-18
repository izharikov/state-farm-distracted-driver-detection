from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import numpy as np


def get_model(summary=False):
    # Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    if summary:
        model_vgg16_conv.summary()

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(150, 150, 3), name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = Model(input=input, output=x)
    if summary:
        my_model.summary()
    return my_model

if __name__=="__main__":
    get_model(True)