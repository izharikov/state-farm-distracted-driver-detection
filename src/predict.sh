#!/usr/bin/env bash
filename=submission-$(date -u +"%m-%d-%Y-%H-%M").csv
model=$1
python main.py --mode predict --path_to_model /content/drive/models/$model --output_file /content/drive/submissions/$filename
# kaggle competitions submit -c state-farm-distracted-driver-detection -f /content/drive/submissions/$filename -m "Message"