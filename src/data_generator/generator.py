from keras.preprocessing.image import ImageDataGenerator

from config import train_dir, validation_dir, test_dir


def get_train_datagen(img_width, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # this is the generator that will read images found in sub-folders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_width),
                                                        batch_size=batch_size, class_mode='categorical', shuffle=True)
    return train_generator


def get_validation_datagen(img_width, batch_size=32):
    # validation image is scaled by 1/255, no other augmentation on validation data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(img_width, img_width),
                                                            batch_size=batch_size,
                                                            class_mode='categorical', shuffle = True)
    return validation_generator


def get_test_datagen(img_width):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_width),
                                                            shuffle=False, class_mode=None, classes=[''])
    return validation_generator
