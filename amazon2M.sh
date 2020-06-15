#!/bin/bash

export METIS_DLL=~/.local/lib/libmetis.so

python main.py --dataset amazon2M --exp_num 1 --batch_size 20 --num_clusters_train 10000 --num_clusters_test 1 --layers 4 --epochs 200 --lr 0.01 --hidden 2048 --dropout 0.2 --lr_scheduler -1 --test 1

