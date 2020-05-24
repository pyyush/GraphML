import time
import torch
import sklearn
import argparse
import numpy as np
import utils as utils
from models import GCN
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', type=str, help='Experiment number for tensorboard')
parser.add_argument('--dataset', , type=str, default='amazon2M', help='options - amazon2M or ppi or reddit')
parser.add_argument('--batch-in-cluster', type=int, default=10)
parser.add_argument('--num-clusters-train', type=int, default=15000)
parser.add_argument('--num-clusters-test', type=int, default=1)
parser.add_argument('--layers', type=int, help='Number of layers in the network.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--fan_in', type=list, help='List of input dim for each layer')
parser.add_argument('--fan_out', type=list, help='List of output dim for each layer')
parser.add_argument('--hidden', type=int, default=400, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (ratio of units to drop).')
#parser.add_argument('--init', type=str, default='glorot_uniform', help='Initialization to use.')
parser.add_argument('--act', type=str, default='relu', help='Activation function to use.')
#parser.add_argument('--optim', type=str, default='Adam', help='Optimization algorithm to use.')
args = parser.parse_args()


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Tensorboard Writer
writer = SummaryWriter('runs/exp' + args.exp_num)


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


@torch.no_grad()
def test(model, features, adj, labels, mask):
        model.cpu()
        model.eval()
        
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0]).t(), torch.FloatTensor(adj[1]), adj[2])
            
        output = model(adj, features)
        
        pred = output[mask].argmax(dim=1, keepdim=True)
        labels = torch.max(labels[mask], 1)[1]
        f1 = sklearn.metrics.f1_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='micro')

        return f1


def main():
    
    start = time.time()

    # Load data
    num_nodes, train_adj, train_feats, labels, train_nodes, full_adj, full_feats, test_nodes = utils.load_data(args.dataset)
    visible_data = train_nodes
    y_train = labels[train_nodes]
    y_test = np.zeros(labels.shape)
    y_test[test_nodes, :] = labels[test_nodes, :]
    test_mask = utils.sample_mask(test_nodes, labels.shape[0])
    
    # Partition
    _, parts = utils.partition_graph(train_adj, visible_data, args.num_clusters_train)
    parts = [np.array(pt) for pt in parts]

    # Preprocess multicluster train data
    train_features_batches, train_support_batches, y_train_batches = utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, args.num_clusters_train, args.batch_in_cluster)    
    
    # Preprocess Test
    _, test_features, test_support, y_test, test_mask = utils.preprocess(full_adj, full_feats, y_test, test_mask, np.arange(num_nodes), args.num_clusters_test) 
    
    print('Pre-Processing finished in {:.2f} minutes!'.format((time.time() - start) / 60))
    
    # model
    model = GCN(fan_in=args.fan_in, fan_out=args.fan_out, layers=args.layers, dropout=args.dropout, normalize=True, bias=False).float()
    model.cuda()

    # Loss Function
    criterion = torch.nn.NLLLoss()
    
    # Optimization Algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    # Learning Rate Schedule    
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=2000, epochs=args.epochs, anneal_strategy='linear')
    model.train()

    # Start Training
    for epoch in range(args.epochs):
        start = time.time()
        np.random.shuffle(parts)
        avg_loss = 0
        avg_micro = 0
        
        for batch in range(len(train_features_batches)):
            features_b = torch.from_numpy(train_features_batches[batch])
            support_b = train_support_batches[batch]
            y_train_b = y_train_batches[batch]

            loss, micro = train(model, criterion, optimizer, features_b, support_b, y_train_b)
            #scheduler.step()
            avg_loss += loss.item()
            avg_micro += micro
        
        
        # Write loss & accuracy to tensorboard
        writer.add_scalar('Time/train', time.time() - start, epoch)
        writer.add_scalar('Loss/train', avg_loss/len(train_features_batches), epoch)
        writer.add_scalar('Accuracy/train', avg_micro/len(train_features_batches), epoch)
        
        
        # Test Model    
        if epoch%25 == 0 or epoch == args.epochs-1:
            micro = test(model, test_features, test_support, y_test, test_mask)
            writer.add_scalar('Accuracy/test', micro, epoch)
        
        
if __name__ == '__main__':
    main()
