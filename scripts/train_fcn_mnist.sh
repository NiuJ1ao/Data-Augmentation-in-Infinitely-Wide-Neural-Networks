#!/bin/bash

python train_nn.py --model fcn \
 --epochs 100 --batch-size 128 \
 --lr 1 --momentum 0.9