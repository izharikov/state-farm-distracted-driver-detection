import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img

if __name__ == "__main__":
    img = load_img("/home/igor/university/diploma/state-farm-distracted-driver-detection/data/test/img_35409.jpg", target_size=(299, 299))
    prep_img = keras.applications.xception.preprocess_input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = prep_img(x)
