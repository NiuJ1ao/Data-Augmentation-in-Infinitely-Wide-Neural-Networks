#!/bin/bash

python train.py --model resfcn \
 --epochs 10 --batch-size 128 \
 --lr 0.001 --momentum 0.9