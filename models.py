import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import MessagePassing

class serverGIN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
class GINConv2(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv2, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                       torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # print(edge_attr.shape)
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))
        #self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, 6))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x2 = x
        x1 = global_add_pool(x, batch)
        x = self.post(x1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x, x1, x2

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, dropout=0.5):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout
        self.node_encoder = torch.nn.Embedding(1, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layer):
            self.convs.append(GINConv2(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        
        h_list = [self.node_encoder(x).to(torch.float32)]
        
        for layer in range(self.num_layer):
            # print(layer)
            
            h = self.convs[layer](h_list[layer].to(torch.float32), edge_index, edge_attr.to(torch.float32))
            
            h = self.batch_norms[layer](h)
            

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)


            h_list.append(h)
        node_representation = h_list[-1]
        # print(node_representation.shape)
        return node_representation



class ogbGIN(torch.nn.Module):
    def __init__(self, nclass, nhid=300, nlayer=5, dropout=0.5):
        super(ogbGIN, self).__init__()
        self.dropout = dropout
        self.nlayer = nlayer
        self.nhid = nhid
        self.nclass = nclass
        self.gnn_node = GNN_node(self.nlayer, self.nhid)
        self.graph_pred = torch.nn.Linear(self.nhid, self.nclass)
    def forward(self, batched_data):
        # print(batched_data)
        h_node = self.gnn_node(batched_data)
        x2 = h_node
        x1 = global_add_pool(h_node, batched_data.batch)
        x = self.graph_pred(x1)
        x = F.log_softmax(x, dim=1)
        # print(x.shape)
        return x, x1, x2
    def loss(self, pred, label):
        # print(pred.shape)
        # print(label.shape)
        return F.nll_loss(pred.to(torch.float32), label.view(-1,))













