from keras.applications import VGG19
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width = 150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    vgg_19_model = VGG19(weights='imagenet', include_top=False)
    if summary:
        vgg_19_model.summary()

    for layer in vgg_19_model.layers:
        layer.trainable = False

    input_shape = Input(shape=(img_width, img_width, 3), name='image_input')

    # Use the generated model
    output_vgg16_conv = vgg_19_model(input_shape)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = add_fc_layers(x, fc_layers, fc_dropout_layers)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = Model(input=input_shape, output=x)
    if summary:
        my_model.summary()
    return my_model

if __name__=="__main__":
    get_model(True, 224)