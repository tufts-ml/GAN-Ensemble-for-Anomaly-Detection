#!/bin/bash
python train.py --name egbad_mnist_en --abnormal_class 3 --setting egbad --dataset mnist --n_G 3 --n_D 3 --isize 32 --nc 1
