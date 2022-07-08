#!/bin/bash

python train_nn.py --model cnn \
 --epochs 100 --batch-size 256 \
 --lr 1 --momentum 0.9