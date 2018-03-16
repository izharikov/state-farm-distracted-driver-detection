#!/bin/bash 
pip install keras
pip install kaggle
apt-get install tree
git clone https://github.com/izharikov/state-farm-distracted-driver-detection
cd state-farm-distracted-driver-detection
mkdir data && cd data
kaggle competitions download -c state-farm-distracted-driver-detection -p .
mv state-farm-distracted-driver-detection/* . && rm -rf state-farm-distracted-driver-detection
unzip imgs.zip > /dev/null && rm imgs.zip
cd ../src
./splitter.sh
cd ..
tree -L 3 -I test
echo Colab configured
