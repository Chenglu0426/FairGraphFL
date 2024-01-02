import random
from random import choices
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import UPFD
from models import GIN, serverGIN, newsModel, serverNewsModel, ogbGIN
from server import Server
from client import Client_GC, Motif_graph
from utils import get_maxDegree, get_stats, split_data, get_numGraphLabels
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import ToUndirected
from sklearn.cluster import KMeans
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class GenData(object):
    def __init__(self, g_list, node_labels, graph_labels):
        self.g_list = g_list
        self.node_labels = node_labels
        self.graph_labels = graph_labels




def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))





    return graphs_chunks

def fakechunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    
    k = num_client
    features = []
    graphs_chunks = [[]for _ in range(k)]
    for i, graph in enumerate(graphs):
        feature = graph.x[0]
        if i == 0:
            features = feature.unsqueeze(0)
        else:
            features = torch.cat((features, feature.unsqueeze(0)), dim=0)
    features = features.cpu().numpy()
    k = num_client
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(features)
    labels = kmeans.labels_
    for i, label in enumerate(labels):
        graphs_chunks[label].append(graphs[i])
    return graphs_chunks


    
def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data



def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, aug=False):
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    elif data == 'fakenews':
        train_set = UPFD(f"{datapath}/UPFD", 'gossipcop', 'content', 'train', ToUndirected())
        test_set = UPFD(f"{datapath}/UPFD", 'gossipcop', 'content', 'test', ToUndirected())
        val_set = UPFD(f"{datapath}/UPFD", 'gossipcop', 'content', 'val', ToUndirected())

    elif data == 'ogb':
        tudataset = PygGraphPropPredDataset(name = 'ogbg-ppa', transform=add_zeros)
        # print(tudataset[0])
        # split_idx = tudataset.get_idx_split()
        # ogd_train = tudataset[split_idx["train"]]
        # ogd_val = tudataset[split_idx["valid"]]
        # ogd_test = tudataset[split_idx["test"]]
        # graphs_train = [x for x in ogd_train]
        # graphs_val = [x for x in ogd_val]
        # graphs_test = [x for x in ogd_test]
        # num_node_features = graphs_train[0].num_node_features

        # ogdtrain_graph_chunks = _randChunk(graphs_train, num_client, overlap, seed=seed)
        # ogdval_graph_chunks = _randChunk(graphs_val, num_client, overlap, seed=seed)
        # ogdtest_graph_chunks = _randChunk(graphs_test, num_client, overlap, seed=seed)
        # graphs_chunks = (ogdtrain_graph_chunks, ogdval_graph_chunks, ogdtest_graph_chunks)

        # for idx, chunks in enumerate(graphs_chunks):
        #     ds = f'{idx}-{data}'
        #     ds_tvt = chunks
        #     ds_train, ds_val, ds_test = chunks[0], chunks[1], chunks[2]
        #     dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        #     dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=False)
        #     dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=False)
        #     num_graph_labels = get_numGraphLabels(ds_train)
        #     splitedData = {}
        #     df = pd.DataFrame()
        #     splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
        #                     num_node_features, num_graph_labels, len(ds_train), ds_train)
        #     df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)
        #     return splitedData, df


    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    if data != 'fakenews':
        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))
    else:
        graphs = [x for x in train_set] + [x for x in test_set] + [x for x in val_set]
        print("  **", data, len(graphs))
        # print(graphs[0].is_directed())
    if data != 'fakenews':
        graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    else:
        graphs_chunks = fakechunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    #print(graphs_chunks)
    if aug:
        aug_rate = []
        for i in range(num_client):
            aug_rate.append(random.uniform(0, 1))
        print(aug_rate)
    for idx, chunks in enumerate(graphs_chunks):
        print(len(chunks))
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        if aug:
            for graph in ds_train:
                node_num, _ = graph.x.size()
                _, edge_num = graph.edge_index.size()
                permute_num = int(edge_num * aug_rate[idx])
                edge_index = graph.edge_index.numpy()
                
                idx_add = np.random.choice(node_num, (2, permute_num))

                edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
                graph.edge_index = torch.tensor(edge_index)

        
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
       
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train), ds_train)
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df

def prepareData_multiDS(datapath, group='small', batchSize=32, convert_x=False, seed=None):
    assert group in ['molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny", 'fakenews']

    if group == 'molecules' or group == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if group == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",   # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                                # bioinformatics
        # datasets = ["MUTAG",                  # small molecules
        #               'ENZYMES']                                # bioinformatics
    
    if group == 'mix' or group == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",   # small molecules
                    "ENZYMES", "DD", "PROTEINS",                                # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                      # social networks
    if group == 'biochem' or group == 'biochem_tiny':
        datasets = ["ENZYMES", "DD", "PROTEINS"]  
    if group == 'fakenews':
        datasets = ['politifact', 'gossipcop']                             # bioinformatics

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data != 'politifact' and data != 'gossipcop':
            # if data == "COLLAB":
            #     tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
            # elif data == "IMDB-BINARY":
            #     tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
            # elif data == "IMDB-MULTI":
            #     tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
            
            # else:
            #     tudataset = TUDataset(f"{datapath}/TUDataset", data)
            #     if convert_x:
            #         maxdegree = get_maxDegree(tudataset)
            #         tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))

            # graphs = [x for x in tudataset]
            # print("  **", data, len(graphs))

            # graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            # graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
            # if group.endswith('tiny'):
            #     graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            #     graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            #     graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

            # num_node_features = graphs[0].num_node_features
            # num_graph_labels = get_numGraphLabels(graphs_train)

            # dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
            # dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
            # dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

            # splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
            #                     num_node_features, num_graph_labels, len(graphs_train))

            # df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
            pass
        if data == 'politifact' or data == 'gossipcop':
        
            train_dataset = UPFD(f"{datapath}/UPFD", data, 'content', 'train', ToUndirected())

            graphs_train = [x for x in train_dataset]
            print("  **", data, len(graphs_train))
            num_node_features = graphs_train[0].num_node_features
            num_graph_labels = get_numGraphLabels(train_dataset)

            val_dataset = UPFD(f'{datapath}/UPFD', data, 'content', 'val', ToUndirected())
            graphs_val = [x for x in val_dataset]

            test_dataset = UPFD(f'{datapath}/UPFD', data, 'content', 'test', ToUndirected())
            graphs_test = [x for x in test_dataset]

            dataloader_train = DataLoader(train_dataset, batch_size=128, shuffle=True)
            dataloader_val = DataLoader(val_dataset, batch_size=128, shuffle=False)
            dataloader_test = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                                num_node_features, num_graph_labels, len(graphs_train), graphs_train)
            df = get_stats(df, data, graphs_train, graphs_val = graphs_val, graphs_test = graphs_test)

    return splitedData, df


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size, graphs_train = splitedData[ds]
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        if args.data_group == 'fakenews':
            cmodel_gc = newsModel(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        if args.data_group == 'ogb':
            cmodel_gc = ogbGIN(num_graph_labels, args.hidden, args.nlayer, args.dropout)
        # optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, graphs_train, dataloaders, optimizer, args))

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    if args.data_group == 'fakenews':
        smodel = newsModel(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        smodel = serverNewsModel(num_node_features, args.hidden)
    if args.data_group == 'ogb':
        smodel = ogbGIN(num_graph_labels, args.hidden, args.nlayer, args.dropout)
    # smodel = newsModel(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)  
    # smodel = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
    server = Server(smodel, args.device)
    return clients, server, idx_clients
