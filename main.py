# Author: Piyush Vyas

import time
import torch
import argparse
import numpy as np
import utils as utils
from models import GCN
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='options - amazon2M or ppi')
parser.add_argument('--exp_num', type=str, help='experiment number for tensorboard')
parser.add_argument('--test', type=int, default=-1, help='True if 1, else False')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_clusters_train', type=int, default=15000)
parser.add_argument('--num_clusters_val', type=int, default=2)
parser.add_argument('--num_clusters_test', type=int, default=1)
parser.add_argument('--layers', type=int, default=4, help='Number of layers in the network.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lr_scheduler', type=int, default=-1, help='True if 1, else False')
parser.add_argument('--hidden', type=int, default=400, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (ratio of units to drop).')
args = parser.parse_args()


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Tensorboard Writer
writer = SummaryWriter('runs/' + args.dataset + '/' + args.exp_num)

# Settings based on ClusterGCN's Table 4. of section Experiments
if args.dataset == 'amazon2M' and args.layers == 4: 
    _in = [100, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, 47]
    
if args.dataset == 'ppi' and args.layers == 5:
    _in = [50, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, args.hidden, 121]


def train(model, criterion, optimizer, features, adj, labels, dataset):
        optimizer.zero_grad()
        
        features = torch.from_numpy(features).cuda()
        labels = torch.LongTensor(labels).cuda()
        
        # Adj -> Torch Sparse Tensor
        i = torch.LongTensor(adj[0]) # indices
        v = torch.FloatTensor(adj[1]) # values
        adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).cuda()
        
        output = model(adj, features)
        
        if dataset == 'ppi':
            loss = criterion(output, labels.type_as(output))
        else:    
            loss = criterion(output, torch.max(labels, 1)[1])

        loss.backward()
        optimizer.step()
    
        return loss


@torch.no_grad()
def test(model, features, adj, labels, mask, device, dataset):
        model.eval()
        
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)
        
        if device == 'cpu':
            adj = adj[0]
            features = features[0]
            labels = labels[0]
            mask = mask[0]
            
        # Adj -> Torch Sparse Tensor
        i = torch.LongTensor(adj[0]) # indices
        v = torch.FloatTensor(adj[1]) # values
        adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).to(device)
        
        output = model(adj, features)
        
        if dataset == 'ppi':
            f1 = utils.f1_score(labels[mask].cpu().numpy(), output[mask].cpu().numpy(), multitask=True)
        else:
            pred = output[mask].argmax(dim=1, keepdim=True)
            labels = torch.max(labels[mask], 1)[1]
            f1 = utils.f1_score(labels.cpu().numpy(), pred.cpu().numpy())

        return f1


def main():
    
    # Load data
    start = time.time()
    N, _adj, _feats, _labels, train_adj, train_feats, train_nodes, val_nodes, test_nodes, y_train, y_val, y_test, val_mask, test_mask = utils.load_data(args.dataset)
    print('Loaded data in {:.2f} seconds!'.format(time.time() - start))
    
    # Prepare Train Data
    start = time.time()
    if args.batch_size > 1: # amazon2M will go here
        _, parts = utils.partition_graph(train_adj, train_nodes, args.num_clusters_train)
        parts = [np.array(pt) for pt in parts]
        train_features, train_support, y_train = utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, args.num_clusters_train, args.batch_size)    
    else: # ppi will go here
        parts, train_features, train_support, y_train, _ = utils.preprocess(train_adj, train_feats, y_train, train_nodes, args.num_clusters_train)
    print('Train Data pre-processed in {:.2f} seconds!'.format(time.time() - start))
    
    # Prepare Val Data
    if args.dataset == 'ppi':
        start = time.time()
        _, val_features, val_support, y_val, val_mask = utils.preprocess(_adj, _feats, y_val, np.arange(N), args.num_clusters_val, val_mask)
        print('Val Data pre-processed in {:.2f} seconds!'.format(time.time() - start))
    
    # Prepare Test Data
    if args.test == 1:    
        if args.dataset == 'amazon2M': # amazon2M val is test
            y_test, test_mask = y_val, val_mask
        start = time.time()
        _, test_features, test_support, y_test, test_mask = utils.preprocess(_adj, _feats, y_test, np.arange(N), args.num_clusters_test, test_mask) 
        print('Test Data pre-processed in {:.2f} seconds!'.format(time.time() - start))
    
    batch_idxs = list(range(len(train_features)))
    
    # model
    model = GCN(fan_in=_in, fan_out=_out, layers=args.layers, dropout=args.dropout, normalize=True, bias=False).float()
    model.cuda()

    # Loss Function
    if args.dataset == 'ppi': # ppi is MultiLabel
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Optimization Algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    # Learning Rate Schedule    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=int(args.num_clusters_train/args.batch_size), epochs=args.epochs+1, anneal_strategy='linear')
    model.train()

    
    # Train on gpu
    for epoch in range(args.epochs + 1):
        np.random.shuffle(batch_idxs)
        avg_loss = 0
        start = time.time()
        for batch in batch_idxs:
            loss = train(model.cuda(), criterion, optimizer, train_features[batch], train_support[batch], y_train[batch], dataset=args.dataset)
            if args.lr_scheduler == 1:
                scheduler.step()
            avg_loss += loss.item()
        
        # Write Train stats to tensorboard
        writer.add_scalar('time/train', time.time() - start, epoch)
        writer.add_scalar('loss/train', avg_loss/len(train_features), epoch)
        
        # Validate on gpu
        if args.dataset == 'ppi':
            avg_f1 = 0
            start = time.time()
            for batch in range(len(val_features)):
                f1 = test(model, val_features[batch], val_support[batch], y_val[batch], val_mask[batch], device='cuda', dataset=args.dataset)
                avg_f1 += f1
                
            # Write Validation stats to tensorboard
            writer.add_scalar('time/val', time.time() - start, epoch)
            writer.add_scalar('f1/val', avg_f1/len(val_features), epoch)
            
    if args.test == 1:    
        # Test on cpu
        f1 = test(model.cpu(), test_features, test_support, y_test, test_mask, device='cpu', dataset=args.dataset)
        print('f1: {:.4f}'.format(f1))
     
        
if __name__ == '__main__':
    main()
