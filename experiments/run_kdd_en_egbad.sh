#!/bin/bash
python train.py --name egbad_kdd_en --abnormal_class 3 --setting egbad --dataset KDD99 --n_G 3 --n_D 3 --nz 32 --lr 0.002