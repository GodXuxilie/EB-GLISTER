# EB-GLISTER
CS6217 Project --- Embedding-Boosted GLISTER

Our code is based on [cords](https://github.com/decile-team/cords.git). We modified the [glisterstrategy.py](https://github.com/GodXuxilie/EB-GLISTER/blob/caecc4f1bae65aea09afef1733a1ae79c2538179/cords/selectionstrategies/SL/glisterstrategy.py) where we incorporated the gradients of word embeddings into the calculation of gain, and [dataselectionstrategy.py](https://github.com/GodXuxilie/EB-GLISTER/blob/caecc4f1bae65aea09afef1733a1ae79c2538179/cords/selectionstrategies/SL/dataselectionstrategy.py) where we initialized the gradients of word embeddings at the beginning of subset selection.

## Requirement
PyTorch 1.11

## How to use?
Just ```bash python train.py```

Noted: run.sh provides all the scipts!
