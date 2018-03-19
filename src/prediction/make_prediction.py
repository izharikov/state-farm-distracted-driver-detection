from data_generator.generator import get_test_datagen
import numpy as np

from models import get_model


def load_model(path_to_model, model_type, img_width):
    model = get_model(modelType=model_type, print_summary=False, img_width=img_width)
    model.load_weights(path_to_model)
    return model


def make_prediction(path_to_model, output_file_csv, model_type, steps=None, img_width=150):
    model = load_model(path_to_model, model_type, img_width=img_width)
    generator = get_test_datagen(img_width)
    result = model.predict_generator(generator=generator, verbose=1, workers=8, use_multiprocessing=True,
                                     steps=steps)
    filenames = generator.filenames
    if (steps is not None):
        filenames = filenames[0:steps * generator.batch_size]
    f = open(output_file_csv, 'wt')
    f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    for i in range(0, len(filenames)):
        f.write("{0}, {1}\n".format(filenames[i], np.array2string(result[i], max_line_width=1000, separator=',')[1:-1]))
    f.close()
    return result
