#!/bin/bash

method=random
model=fcn

nums=$(python -c "import numpy as np; print(' '.join([str(round(i)) for i in np.arange(500, 10500, 500)]))")
for num in ${nums[@]}
do
python isnngp_inference.py --model $model --select-method $method \
--batch-size 0 --device-count -1 --num-inducing-points $num \
--augment-X /vol/bitbucket/yn621/data/mnist70k-augs-patterns.npy --augment-y /vol/bitbucket/yn621/data/mnist70k-augs-labels.npy
done