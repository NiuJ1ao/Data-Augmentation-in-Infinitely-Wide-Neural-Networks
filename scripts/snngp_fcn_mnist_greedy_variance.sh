#!/bin/bash

M=10

nohup python snngp_greedy_variance.py --model fcn \
--batch-size 0 --device-count -1 --num-inducing-points $M \
> log/greedy_variance_fcn_mnist_$M.log 2>&1 &

tail -f log/greedy_variance_fcn_mnist_$M.log