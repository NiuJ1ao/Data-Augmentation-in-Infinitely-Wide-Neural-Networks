#!/bin/bash

python train_nn.py --model resnet \
 --epochs 10 --batch-size 256 \
 --lr 1 --momentum 0.9