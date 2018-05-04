from keras.applications import VGG19, DenseNet121
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, regularizers
from keras.models import Model
from keras_contrib.applications.densenet import DenseNetImageNet121

from models.fc_layers import add_fc_layers


def get_model(summary=False, img_width=150, fc_layers=[4096, 4096], fc_dropout_layers=[0.5, 0.5]):
    # Get back the convolutional part of a VGG network trained on ImageNet
    base_model = DenseNet121(input_tensor=Input(shape=(img_width, img_width, 3)), include_top=False)
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)
    my_model = Model(input=base_model.input, output=x)
    for i in range(141):
        my_model.layers[i].trainable = False
    if summary:
        print("---------------------------------------------------------")
        for i, layer in enumerate(my_model.layers):
            print(i, layer.name)
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        my_model.summary()
    return my_model

if __name__ == "__main__":
    model = get_model(True, 224)
    print('Layers', len(model.layers))
