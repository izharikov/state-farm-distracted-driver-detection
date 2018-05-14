import numpy as np
import pandas as pd
import os
import time

config = {
    'base_path': '~/Downloads',
    'files': [
        {'name': 'submission-from-inception_v3-06-acc-0.9429-loss-0.2112.hdf5-02-05.csv',
         'coeff': 0.45
         },
        {'name': 'submission-from-resnet50-05-acc-0.8765-loss-0.4305.hdf5-03-05.csv',
         'coeff': 0.21
         },
        {'name': 'submission-from-inception_v4-03-acc-0.9489-loss-1.0869.hdf5-06-05.csv',
         'coeff': 0.29
         },
        {'name': 'submission-from-xception-02-acc-0.9573-loss-0.3104.hdf5-06-05.csv',
         'coeff': 0.05
         }
    ],
    'save_path': '/home/igor/university/diploma/submisions'
}

base_path = config['base_path']
files = config['files']

result = pd.read_csv(os.path.join(base_path, files[0]['name']))
result = result[['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']].values
final_result = {}

for res in result:
    final_result[res[0]] = res[1:] * files[0]['coeff']

for file in files[1:]:
    file_res = pd.read_csv(os.path.join(base_path, file['name']))
    file_res = file_res[['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']].values
    for res in file_res:
        final_result[res[0]] += (res[1:] * file['coeff'])


f = open(os.path.join(config['save_path'], 'submission-{0}.csv'.format(time.strftime("%d-%m-%H-%M-%S"))), 'wt')
f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
for file_name, res in final_result.items():
    f.write("{0}, {1}\n".format(file_name, np.array2string(res, max_line_width=1000, separator=',')[1:-1]))
f.close()

