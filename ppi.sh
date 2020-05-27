#!/bin/bash

export METIS_DLL=~/.local/lib/libmetis.so

python main.py --dataset ppi --exp_num 1 --batch_size 1 --num_clusters_train 50 --num_clusters_val 2 --num_clusters_test 1 --layers 5 --epochs 400 --lr 0.01 --hidden 512 --dropout 0.2 --lr_scheduler -1 --test 1

