from keras.applications import Xception
from keras.layers import Input, Flatten, Dense, Dropout, regularizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    x = Input((img_width, img_width, 3))
    base_model = Xception(input_tensor=x, weights='imagenet', include_top=False)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(base_model.input, x)

    for i in range(80):
        model.layers[i].trainable = False
    if summary:
        model.summary()
    return model


if __name__ == "__main__":
    get_model(True, 299)
