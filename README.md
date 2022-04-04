# EB-GLISTER
CS6217 Project --- Embedding-Boosted GLISTER (EB-GLISTER)

Our subset selection code is based on [cords](https://github.com/decile-team/cords.git). We modified the [glisterstrategy.py](https://github.com/GodXuxilie/EB-GLISTER/blob/caecc4f1bae65aea09afef1733a1ae79c2538179/cords/selectionstrategies/SL/glisterstrategy.py) where we incorporated the gradients of word embeddings into the calculation of gain, and [dataselectionstrategy.py](https://github.com/GodXuxilie/EB-GLISTER/blob/caecc4f1bae65aea09afef1733a1ae79c2538179/cords/selectionstrategies/SL/dataselectionstrategy.py) where we initialized the gradients of word embeddings at the beginning of subset selection. To mitigate the heavy usage of GPU memory, we alternatively access and save the gradients on CPUs and GPUs.

## Requirement
PyTorch 1.11

## How to accelerate NLP training?
Before training, please download GloVe weights and nltk\_data directories from this [link](https://drive.google.com/drive/folders/107BLQbg25RWHk922Uq6RUjWi0Iy0UjaF?usp=sharing), and then move these two directories into EB-GLISTER directory. <br/>
Then,  ```python train.py``` .

Noted: run.sh provides all the scipts!

## Collaborators
[Xilie Xu](https://github.com/GodXuxilie) / [Rui Qiao](https://github.com/qiaoruiyt) / [Xiaoqiang Lin](https://xqlin98.github.io)<br/>

