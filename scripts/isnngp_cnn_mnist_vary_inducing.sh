#!/bin/bash

method=random
model=cnn

nohup python isnngp_vary_inducing.py --model $model --select-method $method \
--batch-size 2000 --device-count -1 \
--augment-X /vol/bitbucket/yn621/data/infimnist/mnist120k-augs-patterns-idx3-ubyte --augment-y /vol/bitbucket/yn621/data/infimnist/mnist120k-augs-labels-idx1-ubyte \
> log/isnngp_vary_inducing_${method}_${model}_mnist.log 2>&1 &

tail -f log/isnngp_vary_inducing_${method}_${model}_mnist.log