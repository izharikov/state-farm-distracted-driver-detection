from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Dropout, regularizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    inception_v3_model = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_shape=(img_width, img_width, 3))
    # return inception_v3_model

    # Use the generated model
    output_inception_conv = inception_v3_model.output

    # Add the fully-connected layers

    x = GlobalAveragePooling2D(name='avg_pool')(output_inception_conv)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

    # Create your own model
    my_model = Model(input=inception_v3_model.input, output=x)
    for i in range(180):
        model.layers[i].trainable = False
    if summary:
        my_model.summary()
    return my_model


if __name__ == "__main__":
    model = get_model(True, 299)
    print('Length', len(model.layers))
