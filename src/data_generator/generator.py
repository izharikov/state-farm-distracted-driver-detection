from keras.preprocessing.image import ImageDataGenerator

from config import train_dir, validation_dir, test_dir


def get_train_datagen():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # this is the generator that will read images found in sub-folders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                        batch_size=32, class_mode='categorical')
    return train_generator


def get_validation_datagen():
    # validation image is scaled by 1/255, no other augmentation on validation data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                            batch_size=32, class_mode='categorical')
    return validation_generator

def get_test_datagen():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150),
                                                            shuffle=False, class_mode=None, classes=[''])
    return validation_generator