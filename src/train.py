'''
This program is used to detect the driver's status (10 statuses) by using a small convolutional neural network, which
is trained fram scatch using the training images.
'''
from keras.optimizers import Adam, SGD

from config import simple_model_name
from data_generator.generator import get_train_datagen, get_validation_datagen
from main import getopts
from models import get_model
from models.callbacks import get_callbacks


def train(model_type, num_of_epochs, data_set, img_width=150, optimizer_type='adam', print_summary=False,
          batch_size=32, learning_rate=5e-5, weight_path=None):
    model = get_model(model_type, img_width, print_summary=print_summary)
    model_opt = None
    if optimizer_type == 'adam':
        model_opt = Adam(lr=learning_rate)
    if optimizer_type == 'sgd':
        model_opt = SGD(lr=learning_rate, momentum=0.9)
    if weight_path != None:
        model.load_weights(weight_path)
    model.compile(loss='categorical_crossentropy', optimizer=model_opt, metrics=['accuracy'])

    # this is the generator that will read images found in sub-folders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = get_train_datagen(img_width, batch_size)

    # this is the  generator for validation data
    validation_generator = get_validation_datagen(img_width, batch_size)

    if data_set == 'small':
        train_generator = validation_generator

    # train the convolutional neural network
    model.fit_generator(generator=train_generator, epochs=num_of_epochs,
                        validation_data=validation_generator,
                        callbacks=get_callbacks(model_type))

    # save the weights
    # model.save_weights('driver_state_detection_small_CNN.h5')


if __name__ == "__main__":
    opts = getopts()
    model_type = opts.get('--model', 'simple')
    num_of_epochs = int(opts.get('--epochs', '20'))
    data_set = opts.get('--dataset', 'normal')
    width = int(opts.get('--width', '150'))
    optimizer = opts.get('--optimizer', 'adam')
    print_summary = opts.get('--summary', 'False') == 'True'
    batch_size = int(opts.get('--batch', '32'))
    lr = float(opts.get('--lr', '5e-5'))
    weight_path = opts.get('--weight_path', None)
    train(model_type, num_of_epochs, data_set, img_width=width, optimizer_type=optimizer, print_summary=print_summary,
          batch_size=batch_size, learning_rate=lr, weight_path=weight_path)
