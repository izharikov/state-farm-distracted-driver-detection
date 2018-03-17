#!/bin/bash 
pip install keras > /dev/null
pip install kaggle > /dev/null
apt-get install tree
mkdir ../models
mkdir ../data && cd ../data
kaggle competitions download -c state-farm-distracted-driver-detection -p .
mv state-farm-distracted-driver-detection/* . && rm -rf state-farm-distracted-driver-detection
unzip imgs.zip > /dev/null && rm imgs.zip
cd ../src
./splitter.sh
cd ..
pwd
tree -L 3 -I test
echo Colab configured
