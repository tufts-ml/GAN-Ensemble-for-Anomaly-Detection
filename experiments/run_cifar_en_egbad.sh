#!/bin/bash
python train.py --name egbad_cifar_en --abnormal_class 3 --setting egbad --dataset cifar10 --n_G 3 --n_D 3 --isize 32
