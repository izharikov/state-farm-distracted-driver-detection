import os

from keras.callbacks import ModelCheckpoint

from config import models_dir


def get_callbacks(base_name):
    msave = ModelCheckpoint(os.path.join(models_dir, "%s-{epoch:02d}-{val_acc:.2f}.hdf5" % base_name),
                            monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    return [msave]
