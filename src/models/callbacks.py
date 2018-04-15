import os

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from config import models_dir


def learning_rate(epoch):
    ini_lr = 0.002
    lr = ini_lr * pow(10, -epoch)
    return lr


# es = EarlyStopping()
lrs = LearningRateScheduler(learning_rate)


def get_callbacks(base_name):
    msave = ModelCheckpoint(
        # was: %s-{epoch:02d}-acc-{val_acc:.4f}-loss-{val_loss:.4f}.hdf5
        os.path.join(models_dir, "%s-{epoch:02d}.hdf5" % base_name))
        #,
       # monitor='val_acc', verbose=1, mode='min')
    return [msave, lrs]
