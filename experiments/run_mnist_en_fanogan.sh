#!/bin/bash
python train.py --name fanogan_mnist_en --abnormal_class 3 --setting f-anogan --dataset mnist --n_G 3 --n_D 3 --isize 32 --nc 1
