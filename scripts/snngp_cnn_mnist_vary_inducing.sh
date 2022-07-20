#!/bin/bash

method="random"
model="cnn"

nohup python snngp_vary_inducing.py --model $model --batch-size 2000 --device-count -1 --select-method $method > log/vary_inducing_${method}_${model}_mnist.log 2>&1 &
tail -f log/vary_inducing_${method}_${model}_mnist.log