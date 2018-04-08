from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    inception_v3_model = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_shape=(img_width, img_width, 3))
    # return inception_v3_model

    for layer in inception_v3_model.layers:
        layer.trainable = False


    # Use the generated model
    output_inception_conv = inception_v3_model.output

    # Add the fully-connected layers

    x = GlobalAveragePooling2D(name='avg_pool')(output_inception_conv)
    x = add_fc_layers(x, fc_layers, fc_dropout_layers)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = Model(input=inception_v3_model.input, output=x)
    if summary:
        my_model.summary()
    return my_model


if __name__ == "__main__":
    model = get_model(True, 229, fc_layers=[512], fc_dropout_layers=[0.3])
    print('Length', len(model.layers))
