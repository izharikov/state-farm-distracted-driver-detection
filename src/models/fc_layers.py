from keras.layers import Dense, Dropout
from keras import regularizers


def add_fc_layers(x, fc_layers, fc_dropout_layers):
    for i in range(0, len(fc_layers)):
        units = fc_layers[i]
        dropout = fc_dropout_layers[i]
        x = Dense(units, activation='relu')(x)
        if (dropout > 0):
            x = Dropout(dropout)(x)
    return x
