#!/bin/bash
cp ../../drive/Colab\ Notebooks/kaggle.json /content/.kaggle
chmod 600 /content/.kaggle/kaggle.json
echo [INFO] Installing keras and kaggle
pip install keras > /dev/null
pip install kaggle > /dev/null
apt-get install tree > /dev/null
#ln -s ../models ../../drive/Colab\ Notebooks/models
mkdir ../data && cd ../data
echo [INFO] Downloading data for state-farm-distracted-driver-detection
kaggle competitions download -c state-farm-distracted-driver-detection -p .
echo [INFO] Data downloaded
echo [INFO] Move and unzip images data
mv state-farm-distracted-driver-detection/* . && rm -rf state-farm-distracted-driver-detection
unzip imgs.zip > /dev/null && rm imgs.zip
echo [INFO] Data unzipped
cd ../src
echo [INFO] Running splitting data for train and validation
./splitter.sh
cd ..
echo [INFO] Folders structure for $(pwd):
tree -L 3 -I test
echo Colab configured
