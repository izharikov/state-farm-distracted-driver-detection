import random

import cv2
from keras.preprocessing.image import ImageDataGenerator

from config import train_dir, validation_dir, test_dir, data_path
import numpy as np
import os

# divide drivers
unique_list_train = ['p016', 'p021', 'p022', 'p024', 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072', 'p075']
unique_list_valid = ['p002', 'p012', 'p014', 'p015']
# print (unique_list_train, unique_list_valid)

# get index: driver_id, class, image name
index = os.path.join(data_path, 'driver_imgs_list.csv')

# build the driver id dictionary and class dictionary
f = open(index, 'r')
id_dict = dict()
class_dict = dict()
lines = f.readlines()
for line in lines[1:]:
    arr = line.strip().split(',')
    if arr[0] not in id_dict.keys():
        id_dict[arr[0]] = [line]
    else:
        id_dict[arr[0]].append(line)
    if arr[1] not in class_dict.keys():
        class_dict[arr[1]] = [line]
    else:
        class_dict[arr[1]].append(line)
f.close()

# split the train list and valid list by id
train_list = []
valid_list = []
for id in id_dict.keys():
    if id in unique_list_train:
        train_list.extend(id_dict[id])
    elif id in unique_list_valid:
        valid_list.extend(id_dict[id])
random.shuffle(train_list)
random.shuffle(valid_list)

from keras.preprocessing import image


# image rotation
def rotate(x, degree, row_axis=0, col_axis=1, channel_axis=2, fill_mode='wrap', cval=0.):
    theta = np.pi / 180 * degree
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


# image shift
def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='wrap', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


# PCA
def RGB_PCA(images):
    pixels = images.reshape(-1, images.shape[-1])
    m = np.mean(pixels, axis=0)
    pixels -= m
    C = np.cov(pixels, rowvar=False)
    l, v = np.linalg.eig(C)
    idx = np.argsort(l)[::-1]
    v = v[:, idx]
    l = l[idx]
    # print (C.shape, len(l), len(v))
    return l, v


def RGB_variations(image, eig_val, eig_vec):
    a = 0.1 * np.random.randn(3)
    v = np.array([a[0] * eig_val[0], a[1] * eig_val[1], a[2] * eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


# change HSV
def randomHueSaturationValue(image, hue_shift_limit=(-10, 10),
                             sat_shift_limit=(-75, 75),
                             val_shift_limit=(-75, 75), u=0.5):
    if np.random.random() < u:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = h + hue_shift

        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = s + sat_shift

        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = v + val_shift

        img[:, :, 0], img[:, :, 1], img[:, :, 2] = h, s, v
        image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return image


def get_im_cv2_aug(path, img_size):
    img = cv2.imread(path)
    img = np.array(img, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # random rotate

    deg = random.uniform(-10, 10)  # random rotate limit
    img = rotate(img, deg)

    # random shift
    wshift = random.uniform(-0.1, 0.1)
    hshift = random.uniform(-0.1, 0.1)
    img = shift(img, wshift, hshift)

    # change HSV
    # img = randomHueSaturationValue(img)

    # PCA
    # img = img/255.0
    # l, v = RGB_PCA(img)
    # img = RGB_variations(img, l, v)
    # img = img * 255.0

    # reduce size
    img = cv2.resize(img, (img_size, img_size))

    # normalization

    img /= 127.5
    img -= 1.

    return img


def get_im_cv2(path, img_size):
    img = cv2.imread(path)
    img = np.array(img, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reduce size
    img = cv2.resize(img, (img_size, img_size))
    # normalization
    img /= 127.5
    img -= 1.
    # print (img[1:5, 1:5, 0])
    return img


from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit(['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])


def one_hot_encode(x):
    return lb.transform(x)


def train_gen(img_size, batch_size):
    current = 0
    while 1:
        x = []
        y = []
        while len(y) < batch_size:
            line = train_list[current]
            arr = line.strip().split(',')
            path1 = os.path.join(train_dir, str(arr[1]), str(arr[2]))
            img = get_im_cv2_aug(path1, img_size)
            if random.random() > 0.5:
                line2 = random.choice(class_dict[arr[1]])
                bname = line2.strip().split(',')[2]
                path2 = os.path.join(train_dir, str(arr[1]), str(bname))
                img2 = get_im_cv2_aug(path2, img_size)
                left = img[:, :150, :]
                right = img2[:, 150:, :]
                img = np.concatenate((left, right), axis=1)
            x.append(img)
            label = one_hot_encode([str(arr[1])])[0]
            y.append(label)
            current += 1
            if current >= len(train_list):
                current = 0
        x = np.array(x)
        x = x.reshape(batch_size, img_size, img_size, 3)
        y = np.array(y, dtype=np.uint8)
        y = y.reshape(batch_size, 10)

        yield (x, y)


def valid_gen(img_size, batch_size):
    current = 0
    while 1:
        x = []
        y = []
        while len(y) < batch_size:
            line = valid_list[current]
            arr = line.strip().split(',')
            path = os.path.join(train_dir, str(arr[1]), str(arr[2]))
            # print (path)
            img = get_im_cv2(path, img_size)
            x.append(img)
            label = one_hot_encode([str(arr[1])])[0]
            y.append(label)
            current += 1
            if current >= len(valid_list):
                current = 0
        x = np.array(x)
        x = x.reshape(batch_size, img_size, img_size, 3)
        y = np.array(y, dtype=np.uint8)
        y = y.reshape(batch_size, 10)
        yield (x, y)


def get_train_datagen(img_width, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # this is the generator that will read images found in sub-folders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_width),
                                                        batch_size=batch_size, class_mode='categorical', shuffle=True)
    return train_generator


def get_validation_datagen(img_width, batch_size=32):
    # validation image is scaled by 1/255, no other augmentation on validation data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255, featurewise_center=True)
    test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(img_width, img_width),
                                                            batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)
    return validation_generator


def get_test_datagen(img_width):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # this is the  generator for validation data
    validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_width),
                                                            shuffle=False, class_mode=None, classes=[''])
    return validation_generator
