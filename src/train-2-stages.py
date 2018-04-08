'''
This program is used to detect the driver's status (10 statuses) by using a small convolutional neural network, which
is trained fram scatch using the training images.
'''
from keras.optimizers import Adam, SGD, RMSprop

from config import simple_model_name
from data_generator.generator import get_train_datagen, get_validation_datagen
from main import getopts
from models import get_model
from models.callbacks import get_callbacks


def train(model_type, num_of_epochs, top_epochs, img_width=150, optimizer_type='adam', print_summary=False,
          batch_size=32, learning_rate=5e-5, weight_path=None, fc_layers=None, dropout=None):
    model = get_model(model_type, img_width, print_summary=print_summary, fc_layers=fc_layers, dropout=dropout)

    model_opt_first_stage = RMSprop(lr=learning_rate)

    # first stage compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=model_opt_first_stage,
                  metrics=['accuracy'])

    train_generator = get_train_datagen(img_width, batch_size)
    validation_generator = get_validation_datagen(img_width, batch_size)

    model.fit_generator(generator=train_generator, epochs=top_epochs,
                        validation_data=validation_generator,
                        callbacks=get_callbacks(model_type))
    layers_len = int(len(model.layers) * 0.9)
    for layer in model.layers[:layers_len]:
        layer.trainable = False
    for layer in model.layers[layers_len:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        epochs=num_of_epochs,
                        validation_data=validation_generator,
                        callbacks=get_callbacks(model_type))


if __name__ == "__main__":
    opts = getopts()
    model_type = opts.get('--model', 'simple')
    num_of_epochs = int(opts.get('--epochs', '20'))
    top_epochs = int(opts.get('--top_epochs', '5'))
    data_set = opts.get('--dataset', 'normal')
    width = int(opts.get('--width', '150'))
    optimizer = opts.get('--optimizer', 'adam')
    print_summary = opts.get('--summary', 'False') == 'True'
    batch_size = int(opts.get('--batch', '32'))
    lr = float(opts.get('--lr', '5e-5'))
    weight_path = opts.get('--weight_path', None)
    fc_layers = int(opts.get('--fc', 2))
    fc_width = int(opts.get('--fc_dim', 4096))
    dropout = float(opts.get('--dropout', 0.5))
    train(model_type, num_of_epochs=num_of_epochs,
          top_epochs=top_epochs, img_width=width, optimizer_type=optimizer,
          print_summary=print_summary,
          batch_size=batch_size, learning_rate=lr, weight_path=weight_path, fc_layers=[fc_width] * fc_layers,
          dropout=[dropout] * fc_layers)
