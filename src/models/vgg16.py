from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, regularizers
from keras.models import Model
import numpy as np

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    vgg_16_model = VGG16(input_tensor=Input(shape=(img_width, img_width, 3)), weights='imagenet', include_top=False)
    x = Flatten(name='flatten')(vgg_16_model.output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Create your own model
    my_model = Model(input=vgg_16_model.input, output=x)
    for i in range(10):
        my_model.layers[i].trainable = False
    if summary:
        my_model.summary()
    return my_model


if __name__ == "__main__":
    get_model(True, 224)
