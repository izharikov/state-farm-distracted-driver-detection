#!/usr/bin/env bash
#model=$2 # path_to_model param
echo $*
for model in /content/drive/models/*$2*
do 
  filename=submission-from-$(basename $model)-$(date -u +"%d-%m").csv
  echo Writing to file $filename
  python3 main.py --mode predict --path_to_model $model --output_file /content/drive/submissions/$filename $*
done
# kaggle competitions submit -c state-farm-distracted-driver-detection -f /content/drive/submissions/$filename -m "Message"
