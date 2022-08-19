#!/bin/bash

method="random"
model="fcn"

nums=$(python -c "import numpy as np; print(' '.join([str(round(i)) for i in np.arange(1000, 21000, 1000)]))")
for num in ${nums[@]}
do
python snngp_inference.py --model $model --batch-size 0 --device-count -1 --select-method $method --num-inducing-points $num \
--augment-X /vol/bitbucket/yn621/data/mnist10k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist10k-augs-labels.npy
done