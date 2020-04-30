import torch
import sklearn
import argparse
import numpy as np
import utils as utils
from models import GCN
from tqdm import tqdm


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='.')
parser.add_argument('--batch-in-cluster', type=int, default=10)
parser.add_argument('--num-clusters-train', type=int, default=20000)
parser.add_argument('--num-clusters-test', type=int, default=1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--init', type=str, default='kaiming')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=1024, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (ratio of units to drop).')
args = parser.parse_args()


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


def train(model, criterion, optimizer, features, adj, labels):
        model.cuda()
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


def test(model, features, adj, labels, mask):
        model.cpu()
        model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features)
            labels = torch.LongTensor(labels)
            adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0]).t(), torch.FloatTensor(adj[1]), adj[2])
            
            output = model(adj, features)
        
            pred = output[mask].argmax(dim=1, keepdim=True)
            labels = torch.max(labels[mask], 1)[1]
            f1 = sklearn.metrics.f1_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='micro')

        return f1


def main():
    
    # Load data
    num_nodes, train_adj, train_feats, labels, train_nodes, full_adj, full_feats, test_nodes = utils.load_data(args.path)
    visible_data = train_nodes
    y_train = labels[train_nodes]
    y_test = np.zeros(labels.shape)
    y_test[test_nodes, :] = labels[test_nodes, :]
    test_mask = utils.sample_mask(test_nodes, labels.shape[0])

    # Preprocess
    _, test_features, test_support, y_test, test_mask = utils.preprocess(full_adj, full_feats, y_test, test_mask, np.arange(num_nodes), args.num_clusters_test) 
    
    # Partition
    _, parts = utils.partition_graph(train_adj, visible_data, args.num_clusters_train)
    parts = [np.array(pt) for pt in parts]

    # Preprocess multicluster train data
    train_features_batches, train_support_batches, y_train_batches = utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, args.num_clusters_train, args.batch_in_cluster)    
    
    # model
    model = GCN(fan_in=100, nhid=args.hidden, nclass=47, dropout=args.dropout, normalize=args.normalize, init=args.init)
    model.float()

    # Loss & Optim
    criterion = torch.nn.NLLLoss()
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    elif args.optim == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim == 'sgd': 
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
            avg_loss += loss.item()
            avg_micro += micro

        torch.save(model.state_dict(), args.optim + '_' + str(args.hidden) + '_' + 'model.pt')
        print('Epoch ->', epoch, '| Loss ->', avg_loss/len(train_features_batches), '| Train F1-Micro ->', avg_micro/len(train_features_batches))
        
    micro = test(model, test_features, test_support, y_test, test_mask)
    print('Test F1-Micro ->', micro)
        
if __name__ == '__main__':
    main()
