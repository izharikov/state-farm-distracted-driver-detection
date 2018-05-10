from config import test_dir
from data_generator.generator import get_test_datagen, test_gen
import numpy as np
import os

from models import get_model


def load_model(path_to_model, model_type, img_width, fc_layers, dropout):
    model = get_model(modelType=model_type, print_summary=False, img_width=img_width, fc_layers=fc_layers,
                      dropout=dropout)
    if (isinstance(model, tuple)):
        model = model[0]
    model.load_weights(path_to_model)
    return model


num_test_samples = 79726


def make_prediction(path_to_model, output_file_csv, model_type, steps=None, img_width=150, fc_layers=None,
                    dropout=None):
    model = load_model(path_to_model, model_type, img_width=img_width, fc_layers=fc_layers, dropout=dropout)
    test_batch_size = 32
    generator = test_gen(img_width, 32, model_type)
    result = model.predict_generator(generator=generator, verbose=1,
                                     steps=(int(float(num_test_samples) / test_batch_size) + 1))
    filenames = os.listdir(test_dir)
    if (steps is not None):
        filenames = filenames[0:steps * generator.batch_size]
    f = open(output_file_csv, 'wt')
    f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    for i in range(0, len(filenames)):
        f.write("{0}, {1}\n".format(filenames[i], np.array2string(result[i], max_line_width=1000, separator=',')[1:-1]))
    f.close()
    return result
