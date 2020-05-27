# Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
This repository contains PyTorch scripts for training Cluster-GCN[1].

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
  datasets/ppi/ppi-{G.json, feats.npy, class_map.json, id_map.json}
  datasets/amazon2M/amazon2M-{G.json, feats.npy, class_map.json, id_map.json}
  
 * Use the dataset specific shell scripts to run experiments on that dataset.
```
./ppi.sh for ppi dataset
./amazon2M for amazon2M dataset
```
## Results:
|               | PPI         | Reddit    |  Amazon2M  | 
| ------------- |:-----------:|:---------:| ----------:|
| Cluster-GCN   | N/A | N/A | **0.8830** |


## Citations:
[1] Wei-Lin Chiang et al. "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"\
[2] Amazon2M http://manikvarma.org/downloads/XC/XMLRepository.html
[3] PPI http://snap.stanford.edu/graphsage/ppi.zip
[4] Reddit http://snap.stanford.edu/graphsage/reddit.zip
