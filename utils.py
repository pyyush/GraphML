import json
import torch
import time
import metis
import numpy as np
import sklearn.metrics
import scipy.sparse as sp
from networkx.readwrite import json_graph
import sklearn.preprocessing

np.random.seed(0)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(path):
    
    print("Loading data.......")
    start = time.time()
    
    # Load data files
    G = json_graph.node_link_graph(json.load(open("./Amazon2M-G.json")))
    
    id_map = json.load(open("./Amazon2M-id_map.json"))
    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
    
    class_map = json.load(open("./Amazon2M-class_map.json"))
    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}
    
    feats = np.load("./Amazon2M-feats.npy").astype(np.float32)
    
    # Generate edge list
    edges = []
    for edge in G.edges():
        if edge[0] in id_map and edge[1] in id_map:
            edges.append((id_map[edge[0]], id_map[edge[1]]))
    
    num_nodes = len(id_map)
    
    test_nodes = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['val']] or G.nodes[n]['test'], dtype=np.int32)
    
    is_train = np.ones((num_nodes), dtype=np.bool)
    is_train[test_nodes] = False

    train_nodes = np.array([n for n in range(num_nodes) if is_train[n]], dtype=np.int32)
    train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
    test_edges = [(e[0], e[1]) for e in edges if not is_train[e[0]] and not is_train[e[1]]]
    
    edges = np.array(edges, dtype=np.int32)
    train_edges = np.array(train_edges, dtype=np.int32)
    test_edges = np.array(test_edges, dtype=np.int32)
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_nodes, num_classes), dtype=np.float32)
    for k in class_map.keys():
        labels[id_map[k], class_map[k]] = 1
        
    train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])

    train_feats = feats[train_ids]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    full_feats = scaler.transform(feats)
    
    def _construct_adj(e, shape):
        adj = sp.csr_matrix((np.ones((e.shape[0]), dtype=np.float32), (e[:, 0], e[:, 1])), shape=shape)
        adj += adj.transpose()
        return adj

    
    train_adj = _construct_adj(train_edges, (len(train_nodes), len(train_nodes)))
    full_adj = _construct_adj(edges, (num_nodes, num_nodes))

    train_feats = full_feats[train_nodes]
            
    print("Data loaded in ", (time.time() - start)/60, "minutes.")
    return num_nodes, train_adj, train_feats, labels, train_nodes, full_adj, full_feats, test_nodes
    
    
    
def partition_graph(adj, idx_nodes, num_clusters):
    
    """partition a graph by METIS."""
    print("Clustering ....... ")
    start_time = time.time()
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

    print("Clustering done in", (time.time() - start_time)/60, "minutes.")
    return part_adj, parts
    
    
def normalize_adj_diag_enhance(adj, diag_lambda=1):
    
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
    
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj
    
    
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

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
    
    
def preprocess(adj, features, y_train, train_mask, visible_data, num_clusters):

    # Do graph partitioning
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
      y_train_batches.append(y_train) #[pt, :]

      train_pt = []
      for newidx, idx in enumerate(pt):
        if train_mask[idx]:
          train_pt.append(newidx)
      train_mask_batches.append(sample_mask(train_pt, len(pt)))
    
    f, s, y, m = np.array(features_batches[0]), np.array(support_batches[0]), np.array(y_train_batches[0]), np.array(train_mask_batches[0])
    
    return (parts, f, s, y, m)

def preprocess_multicluster(adj, parts, features, y_train, num_clusters, block_size):
    """Generate the batch for multiple clusters."""
    start = time.time()
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

    print("Preprocessing multi cluster done in ", (time.time() - start)/60, "minutes.")
    return features_batches, support_batches, y_train_batches
    
