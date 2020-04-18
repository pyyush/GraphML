import time
import argparse
import numpy as np
from models import GCN
import torch
from tqdm import tqdm
import utils
import sklearn


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='.')
parser.add_argument('--precalc', default=False, type=bool, help='Whether to precompute AX or not.')
parser.add_argument('--batch-in-cluster', default=10)
parser.add_argument('--num-clusters-train', default=20000, type=int)
parser.add_argument('--num-clusters-test', default=1)
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (ratio of units to drop).')
args = parser.parse_args()

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


def train(model, criterion, optimizer, features, adj, labels):
        
        optimizer.zero_grad()
        
        labels = torch.LongTensor(labels).cuda()
        adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0]).t(), torch.FloatTensor(adj[1]), adj[2])
        
        output = model(adj.cuda(), features.cuda())
        loss = criterion(output, torch.max(labels, 1)[1])
        
        pred = output.argmax(dim=1, keepdim=True)
        labels = torch.max(labels, 1)[1]
        f1 = sklearn.metrics.f1_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='micro')

        loss.backward()
        optimizer.step()
    
        return loss, f1


def main():
    
    # Load data
    num_nodes, train_adj, train_feats, labels, train_nodes = utils.load_data(args.path)
    visible_data = train_nodes
    y_train = labels[train_nodes]
    #y_test = labels[test_nodes]

    # Check shapes
    #print('A ->', test_adj.shape, 'X ->', test_feats.shape, 'y', y_test.shape)
    #exit()
    
    # Partitioning
    _, parts = utils.partition_graph(train_adj, visible_data, args.num_clusters_train)
    parts = [np.array(pt) for pt in parts]
    print('Clusters ->', len(parts))

    # Preprocess multicluster train data
    train_features_batches, train_support_batches, y_train_batches = utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, args.num_clusters_train, args.batch_in_cluster)
    print('Batches -> ', len(train_features_batches))    
    
    # model
    model = GCN(fan_in=100, nhid=args.hidden, nclass=47, dropout=args.dropout, normalize=True)
    model.float()
    model.cuda()

    # Loss & Optim
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.train()
    
    for epoch in tqdm(range(args.epochs)):
        np.random.shuffle(parts)
        avg_loss = 0
        avg_micro = 0
        
        for batch in range(len(train_features_batches)):
            features_b = torch.from_numpy(train_features_batches[batch])
            support_b = train_support_batches[batch]
            y_train_b = y_train_batches[batch]
            loss, micro = train(model, criterion, optimizer, features_b, support_b, y_train_b)
            avg_loss += loss
            avg_micro += micro
        print('Epoch:', epoch, '| Loss =', avg_loss.item()/len(train_features_batches), '| F1-Micro =', avg_micro/len(train_features_batches))
        
if __name__ == '__main__':
    main()
