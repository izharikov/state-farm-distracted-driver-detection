# Report
## Overview
This report describes results in achieving **0.18887** *Private Kaggle Score* in [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
## Models trained
|Model name|Epoches|Train score|Train Score|Public Score|Private Score|
|---|---|---|---|---| --|
|Xception |2 |0.3104 |0.9573 |0.36424 |0.31099|
| Resnet-50| 5| 0.4305| 0.8765| 0.33764 |0.28924|
| Inception v3| 5 |0.2112 |0.9429 |0.31895 |0.25761|
| Inception v4 |3 |0.9794 |0.9645 |0.37454 |0.30524 |
|Inception-Resnetv2 |3 |0.4168 |0.9273 | 0.40608 |0.39033|
| Densenet-121 |3 |0.5009| 0.878 |0.41703 |0.39612|
|*Conact 4 best models*|-|-|-|0.22724|0.18887|

As you can see in this table, best result for *single* model is **Inception v4** with Private Score **0.25761**.
But was able to significantly improve this score using just simple linear concatenation of 4 best models. The code of this merging is available [here](src/merger.py).
## References
 - Going deeper with convolutions / arxiv.org — [Electronic resource]. — Mode of access: https://arxiv.org/pdf/1409.4842.pdf 
 - Xception: Deep Learning with Depthwise Separable Convolutions / arxiv.org — [Electronic resource]. — Mode of access: https://arxiv.org/pdf/1610.02357.pdf
