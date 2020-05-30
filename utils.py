import json
import torch
import time
import metis
import numpy as np
import sklearn.metrics
import scipy.sparse as sp
from networkx.readwrite import json_graph
import sklearn.preprocessing
import tensorflow as tf

np.random.seed(0)


def load_data(dataset, path='./datasets'):
    
    # Load data files
    feats = np.load(tf.io.gfile.GFile('{}/{}/{}-feats.npy'.format(path, dataset, dataset), 'rb')).astype(np.float32)
    G = json_graph.node_link_graph(json.load(tf.io.gfile.GFile('{}/{}/{}-G.json'.format(path, dataset, dataset))))
    
    id_map = json.load(tf.io.gfile.GFile('{}/{}/{}-id_map.json'.format(path, dataset, dataset)))
    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
    
    class_map = json.load(tf.io.gfile.GFile('{}/{}/{}-class_map.json'.format(path, dataset, dataset)))
    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}
    
    # Generate edge list
    edges = []
    for edge in G.edges():
        if edge[0] in id_map and edge[1] in id_map:
            edges.append((id_map[edge[0]], id_map[edge[1]]))
    
    # Total Number of Nodes in the Graph
    _nodes = len(id_map)
    
    # Seperate Train, Val, and Test nodes
    val_nodes = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['val']], dtype=np.int32)
    test_nodes = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['test']], dtype=np.int32)
    is_train = np.ones((_nodes), dtype=np.bool)
    is_train[test_nodes] = False
    is_train[val_nodes] = False
    train_nodes = np.array([n for n in range(_nodes) if is_train[n]], dtype=np.int32)
    
    # Train Edges
    train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
    train_edges = np.array(train_edges, dtype=np.int32)
    
    # All Edges in the Graph
    _edges = np.array(edges, dtype=np.int32)
    
    # Generate Labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        _labels = np.zeros((_nodes, num_classes), dtype=np.float32)
        for k in class_map.keys():
            _labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        _labels = np.zeros((_nodes, num_classes), dtype=np.float32)
        for k in class_map.keys():
            _labels[id_map[k], class_map[k]] = 1
        
    train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])

    train_feats = feats[train_ids]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    _feats = scaler.transform(feats)
    
    def _construct_adj(e, shape):
        adj = sp.csr_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=shape)
        adj += adj.transpose()
        return adj

    train_adj = _construct_adj(train_edges, (len(train_nodes), len(train_nodes)))
    _adj = _construct_adj(_edges, (_nodes, _nodes))

    train_feats = _feats[train_nodes]
    
    # Generate Labels
    y_train = _labels[train_nodes]
    y_val = np.zeros(_labels.shape)
    y_test = np.zeros(_labels.shape)
    y_val[val_nodes, :] = _labels[val_nodes, :]
    y_test[test_nodes, :] = _labels[test_nodes, :]
    
    # Generate Masks for Validtion & Testing Data
    val_mask = sample_mask(val_nodes, _labels.shape[0])
    test_mask = sample_mask(test_nodes, _labels.shape[0])

    return _nodes, _adj, _feats, _labels, train_adj, train_feats, train_nodes, val_nodes, test_nodes, y_train, y_val, y_test, val_mask, test_mask
    
    
    
def partition_graph(adj, idx_nodes, num_clusters):
    
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]

    for i in range(num_nodes):
      rows = train_adj_lil[i].rows[0]
      # self-edge needs to be removed for valid format of METIS
      if i in rows:
        rows.remove(i)
      train_adj_lists[i] = rows
      neighbors += rows
      edge_cnt += len(rows)
      neighbor_intervals.append(edge_cnt)
      train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
      _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
      groups = [0] * num_nodes

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]

    for nd_idx in range(num_nodes):
      gp_idx = groups[nd_idx]
      nd_orig_idx = idx_nodes[nd_idx]
      parts[gp_idx].append(nd_orig_idx)
  
      for nb_orig_idx in adj[nd_orig_idx].indices:
        nb_idx = train_ord_map[nb_orig_idx]
        if groups[nb_idx] == gp_idx:
          part_data.append(1)
          part_row.append(nd_orig_idx)
          part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    return part_adj, parts
    
    
def preprocess(adj, features, y_train, visible_data, num_clusters, train_mask=None):
    
    # graph partitioning
    part_adj, parts = partition_graph(adj, visible_data, num_clusters)
    part_adj = normalize_adj_diag_enhance(part_adj)
    parts = [np.array(pt) for pt in parts]

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    
    for pt in parts:
      features_batches.append(features[pt, :])
      now_part = part_adj[pt, :][:, pt]
      total_nnz += now_part.count_nonzero()
      support_batches.append(sparse_to_tuple(now_part))
      y_train_batches.append(y_train[pt, :])##
      
      if train_mask is not None:
          train_pt = []
          for newidx, idx in enumerate(pt):
            if train_mask[idx]:
              train_pt.append(newidx)
          train_mask_batches.append(sample_mask(train_pt, len(pt)))
          return parts, features_batches, support_batches, y_train_batches, train_mask_batches
      else:
          return parts, features_batches, support_batches, y_train_batches, train_mask


def preprocess_multicluster(adj, parts, features, y_train, num_clusters, block_size):
    """ Generate batches for multiple clusters."""
    features_batches = []
    support_batches = []
    y_train_batches = []
    total_nnz = 0
    np.random.shuffle(parts)

    for _, st in enumerate(range(0, num_clusters, block_size)):
      pt = parts[st]
      for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
        pt = np.concatenate((pt, parts[pt_idx]), axis=0)
      features_batches.append(features[pt, :])
      y_train_batches.append(y_train[pt, :])
      support_now = adj[pt, :][:, pt]
      support_batches.append(sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda=1)))
      total_nnz += support_now.count_nonzero()

    return features_batches, support_batches, y_train_batches
    
    
def normalize_adj_diag_enhance(adj, diag_lambda=1):
    
    """ A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A') """
    
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj
    
        
def sparse_to_tuple(sparse_mx):

    def to_tuple(mx):
      if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
      coords = np.vstack((mx.row, mx.col)).transpose()
      values = mx.data
      shape = mx.shape
      return coords, values, shape

    if isinstance(sparse_mx, list):
      for i in range(len(sparse_mx)):
        sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
      sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
    
        
def sample_mask(idx, mat):
    mask = np.zeros(mat)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)    
    
    
def f1_score(y_true, y_pred, multitask=False):
    if multitask:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
        return sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    else:
        return sklearn.metrics.f1_score(y_true, y_pred, average='micro')
        
        
