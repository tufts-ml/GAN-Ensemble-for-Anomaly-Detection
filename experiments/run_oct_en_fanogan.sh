#!/bin/bash
python train.py --name fanogan_oct_en --abnormal_class 3 --setting f-anogan --dataset OCT --n_G 3 --n_D 3 --isize 64 --nz 256 -nc 1
