# Cluster-GCN in PyTorch
[![Arxiv](https://img.shields.io/badge/ArXiv-1905.07953-orange.svg?color=blue&style=plastic)](https://arxiv.org/abs/1905.07953)
[![Download](https://img.shields.io/badge/Download-amazon2M-brightgreen.svg?color=black&style=plastic)](https://drive.google.com/drive/folders/1Tfn-yABlW5JheyYItyRyrMGtmQdYN7wm?usp=sharing)
[![Clones](https://img.shields.io/badge/Clones-42-brightgreen.svg?color=brightgreen&style=plastic)](https://github.com/pyyush/ClusterGCN-amazon2M/blob/master/README.md)\
> Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
> Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh.
> KDD, 2019.
> [[Paper]](https://arxiv.org/abs/1905.07953)
Raw data files used to curate this dataset can be downloaded from http://manikvarma.org/downloads/XC/XMLRepository.html while the processed data files used in this implementation can be downloaded by clicking on the above Download amazon2M badge.

## Requirements:
* install the clustering toolkit metis and other required Python packages.
```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
6) pip install -r requirements.txt
```
## Usage:
* The Scipts assume that the data files are stored in the following structure.
  ```
  ./datasets/amazon2M/amazon2M-{G.json, feats.npy, class_map.json, id_map.json}
  ```
 * Run the below shell script to perform experiments on amazon2M dataset.
```
./amazon2M.sh
tensorboard --logdir=runs --bind_all (optional to visualize training)
```
## Results:
* F1-score **0.8866** (vs Cluster-GCN paper - 0.9041)

* Training Plots
<figure>
  <img src="W&B Chart 6_15_2020, 1_02_55 AM.png"/>
</figure>

<figure>
  <img src="W&B Chart 6_15_2020, 1_07_55 AM.png"/>
</figure>
