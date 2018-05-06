import os.path

train_dir = os.path.abspath("../data/train")
validation_dir = os.path.abspath("../data/validation")
test_dir = os.path.abspath("../data/test")
data_path = os.path.abspath("../data")

models_dir = os.path.abspath("../../drive/models")
simple_model_name = "simple"

normalize_zero = ['xception', 'inception_v3', 'vgg16', 'vgg19', 'densenet', 'densenet121', 'inception_resnet_v2', 'inception_v4', 'resnet152']
normalize_mean = ['resnet50']
