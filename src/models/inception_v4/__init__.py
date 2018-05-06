from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Dropout, regularizers, AveragePooling2D
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from models.fc_layers import add_fc_layers
from models.inception_v4.inception_v4_keras_impl import inception_v4


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    inception_v4_model = inception_v4(weights='imagenet', include_top=False)
    # return inception_v3_model

    # Use the generated model
    output_inception_conv = inception_v4_model.output

    # Add the fully-connected layers

    x = AveragePooling2D((8, 8), padding='valid')(output_inception_conv)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

    # Create your own model
    my_model = Model(input=inception_v4_model.input, output=x)
    for i in range(145):
        my_model.layers[i].trainable = False
    if summary:
        print("---------------------------------------------------------")
        for i, layer in enumerate(my_model.layers):
            print(i, layer.name)
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        my_model.summary()
        my_model.summary()
    return my_model


if __name__ == "__main__":
    model = get_model(True, 299)
    print('Length', len(model.layers))
