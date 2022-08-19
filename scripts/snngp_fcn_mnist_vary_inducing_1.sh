#!/bin/bash

method="random"
model="fcn"
nums=$(python -c "import numpy as np; print(' '.join([str(round(i)) for i in np.logspace(np.log10(10), np.log10(100), 20)]))")
for num in ${nums[@]}
do
python snngp_inference.py --model $model --batch-size 0 --device-count -1 --select-method $method --num-inducing-points $num
done