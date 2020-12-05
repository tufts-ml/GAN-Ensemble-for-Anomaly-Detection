# GAN-Ensemble-for-Anomaly-Detection


This repository contains PyTorch implementation of the following paper: GAN-Ensemble-for-Anomaly-Detection

## 0. Environment Setup
pip install -r requirements.txt

## 1. Table of Contents
- [GAN-Ensemble-for-Anomaly-Detection](#GAN-Ensemble-for-Anomaly-Detection)
  - [1. Table of Contents](#1-table-of-contents)
 
  - [2. Experiment](#2-experiment)
  - [3. Training](#3-training)
   
 



## 2. Experiment
To replicate the results in the paper for MNIST and CIFAR10  dataset, run the following commands:

``` shell
#MNIST
sh experiments/run_mnist_en_fanogan.sh
sh experiments/run_mnist_en_egbad.sh
# CIFAR
sh experiments/run_cifar_en_fanogan.sh
sh experiments/run_cifar_en_egbad.sh
#OCT
sh experiments/run_oct_en_fanogan.sh
#KDD99
sh experiments/run_oct_en_egbad.sh


```

## 3. Training
To list the arguments, run the following command:
```
python train.py -h
```

To train the model on MNIST dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset mnist                                                                \
    --niter <number-of-epochs>                                                     \
    --abnormal_class  <0,1,2,3,4,5,6,7,8,9>                                        \
    --setting <model-name: f-anogan, egbad, ganomaly, skipgan>                     \
    --n_G <number of ensemble generators>                                          \
    --n_D <number of ensemble discriminators>                                      \
```    
   
   
   
         


To train the model on CIFAR10 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset cifar10                                                             \
    --niter <number-of-epochs>                                                    \
    --abnormal_class                                                              \
        <0-9 for :airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck>    \
    --setting <model-name: f-anogan, egbad, ganomaly, skipgan>                     \
    --n_G <number of ensemble generators>                                          \
    --n_D <number of ensemble discriminators>                                      \
         
```

To train the model on OCT dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset OCT                                                                  \
    --niter <number-of-epochs>                                                     \
    --setting <model-name: f-anogan, egbad, ganomaly, skipgan>                     \
    --n_G <number of ensemble generators>                                          \
    --n_D <number of ensemble discriminators>                                      \
         
```

To train the model on KDD99 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset KDD99                                                                \
    --niter <number-of-epochs>                                                     \
    --setting <model-name: f-anogan, egbad>                                        \
    --n_G <number of ensemble generators>                                          \
    --n_D <number of ensemble discriminators>                                      \
         
```

