# State farm distracted driver detection
## Overview
Implementation of [Kaggle driver detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
## CNN Models implemented 
- [simple model](src/models/simple_model.py)
- [vgg16](src/models/vgg16.py)
- [vgg19](src/models/vgg19.py)
- [inception v3](src/models/inception_v3.py)
- [xception](src/models/xception.py)
## Train and evaluate
### Downloading and splitting data
Using `kaggle` API:
```bash
./src/colab_config.sh
```
### Train
```bash
python train.py [...options]
```
#### Example of train:
```bash
python train.py --model vgg16 --width 224 --optimizer adam --lr 1e-5 --batch 16 --epochs 50
```
#### Options
* `--model <model>`<br>
type of model used. `<model>` one of the following: `simple`, `vgg16`, `vgg19`, `inception_v3`, `xception`. Default us `simple.`
* `--epochs <number>`<br>
 `<number>` - number of training epochs. Default is 20.
* `--width <width>`<br>
Width of image, that will be input of model. Images from dataset are resized to this `<width>`. Default is 150.
* `--optimizer <optimizer>`<br>
Optimizer, used in train process. `<optimizer>` one of the following: `adam`, `sgd`, `rmsprop`
* `--summary [True|False]`<br>
Print summary of model. Default is `False`
* `--lr <learning_rate>`<br>
Set learning rate for  `<optimizer>`. Default value is `5e-5`
* `--weight_path <path>`<br>
Path to weights. If specified, weight from file is loaded, if not - weights initialized randomly.
* `--fc <count_of_layers>`<br>
Count of fully connected layers, used in fine-tuning. Default is 2.
* `--fc_dim <dimension>`<br>
Dimension of fully connected layers. Default is 4096.
* `--dropout <dropout>`<br>
Dropout after each fully-connected layer. If `< 0`, than no dropout layers added. Default is 0.5.
* `--batch <batch>`<br>
Batch size in train process. Default is 32.
### Evaluation
Files `predict.sh` and `predict-all.sh`.
#### predict.sh
Second param: file name to saved weights in `/content/drive/models` folder:
```bash
./predict.sh --path_to_model vgg-04-acc-0.9722-loss-0.0997.hdf5 --model vgg16 --width 224
```
#### predict-all.sh
For all saved models in folder `/content/drive/models` make prediction and save to `/content/drive/submissions`:
```bash
./predict-all.sh --model vgg16 --width 224
```
## Environment
Training was running on Colaboratory Google Platform on GPU environment.
