# Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
This repository contains scripts for training Cluster-GCN on the Amazon2M dataset introduced in [1]. 
The Amazon2M dataset has 2M nodes and 61M edges based on co-purchase data from [2].

## Requirements:

* install the clustering toolkit: metis and its Python interface.

```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so

Note: If you run jobs on a cluster you will likely need to do perform step 5) every time you are assigned a new node. 
```

* install required Python packages

```
 pip install -r requirements.txt
```

## Run Experiments:

* After metis and networkx are set up, you can try the scripts.

* Download and store data files in their respective folders.

  For example, the path of amazon2M data files should be: .datasets/amazon2M/amazon2M-{G.json, feats.npy, class_map.json, id_map.json}

## Citations:
[1] Wei-Lin Chiang et al. "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"\
[2] Amazon3M http://manikvarma.org/downloads/XC/XMLRepository.html
