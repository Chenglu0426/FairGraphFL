import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from copy import deepcopy
import math
from numpy import *
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch import nn
import torch.nn.functional as F
class Motif_graph():
    def __init__(self, graph, motif_dict):
        self.graph = graph
        
        self.motif_dict = motif_dict

class MotifDataset(Dataset):
    def __init__(self, dataset):
        self.graph = dataset
    def __len__(self):
        return len(self.graph)
    def __getitem__(self, index):
        graph = self.graph[index].graph
        #label = self.graph[index].graph.y
        motif_dict = self.graph[index].motif_dict
        return graph,  motif_dict
    

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, graphs_train, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.graphs_train = graphs_train
        

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.


        self.motif_vocab = {} 
        self.motif_count = {} 
        self.tf_idf = {}
        self.motif_dataset = []
        self.avg_tf = {}
        self.prototype = {}
        self.prototype_code = {} 
        self.code_motif = {}
        self.motif_code = {} 
        self.motif_dataset = []
        self.motif_dict = []
        self.motifset_dict = []


        self.code_prototype = {} 

        self.simi = {}
        self.rs = {} 
        
      
        


        

    def motif_construction(self):
        self.motif_dataset = []
   

        for graph in self.graphs_train:
            motif_freq = {} 
            label = graph.x
            _, label = torch.max(label, dim=1)
            label = label.tolist()

            if graph.edge_attr is not None:
                graph_net = to_networkx(graph, to_undirected=True, edge_attrs=["edge_attr"])
            else:
                graph_net = to_networkx(graph, to_undirected=True)
            mcb = nx.cycle_basis(graph_net)
            mcb_tuple = [tuple(ele) for ele in mcb]
            

            edges = []
            for e in graph_net.edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))


            for e in edges:

                if 'edge_attr' in graph_net.get_edge_data(e[0], e[1]):
                    weight = graph_net.get_edge_data(e[0], e[1])['edge_attr']
                    weight = weight.index(max(weight))
                else:
                    weight = 1
                
                edge = ((label[e[0]], label[e[1]]), weight)
                c = deepcopy(edge[0])
                weight = deepcopy(edge[1])
                for i in range(2):

                    if (c, weight) in self.motif_vocab:
                        self.motif_vocab[(c, weight)] += 1
                    else:
                        c = (label[e[1]], label[e[0]])
                if (c, weight) not in self.motif_vocab:
                    self.motif_vocab[(c, weight)] = 1

                for i in range(2):
                    if (c, weight) in motif_freq:
                        motif_freq[(c, weight)] += 1
                    else:
                        c = (label[e[1]], label[e[0]])
                if (c, weight) not in motif_freq:
                    motif_freq[(c, weight)] = 1




            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, graph_net))
                #print(weight)
                ring = []
                for i in range(len(m)):
                    
                    ring.append(label[m[i]])
                cycle = (tuple(ring), weight)
                c = deepcopy(cycle[0])
                weight = deepcopy(cycle[1])
                for i in range(len(c)):
                    if (c, weight) in self.motif_vocab:
                        self.motif_vocab[(c, weight)] += 1
                    else:
                        c = self.shift_right(c)
                        weight = self.shift_right(weight)
                if (c, weight) not in self.motif_vocab:
                    self.motif_vocab[(c, weight)] = 1

                for i in range(len(c)):
                    if (c, weight) in motif_freq:
                        motif_freq[(c, weight)] += 1
                    else:
                        c = self.shift_right(c)
                        weight = self.shift_right(weight)
                if (c, weight) not in motif_freq:
                    motif_freq[(c, weight)] = 1
            for motif in motif_freq.keys():
                if motif not in self.motif_count.keys():
                    self.motif_count[motif] = 1
                else:
                    self.motif_count[motif] += 1
            graphs = Motif_graph

            self.motif_dataset.append(graphs(graph, motif_freq))
        
        for motif_graph in self.motif_dataset:

            
            for motif in motif_graph.motif_dict:
                c = motif_graph.motif_dict[motif] 
                if c > 0:
                    M = len(self.motif_dataset) 
                    N = self.motif_count[motif] 
                    tf = c * (math.log((1 + M) / (1 + N)) + 1)

                    if motif not in self.tf_idf:
                        self.tf_idf[motif] = []
                        self.tf_idf[motif].append(tf)
                    else:
                        self.tf_idf[motif].append(tf)
        for motif in self.tf_idf.keys():
            self.avg_tf[motif] = mean(self.tf_idf[motif])
        self.avg_tf = sorted(self.avg_tf.items(), key = lambda x: x[1], reverse=True)
        rank_list = []
        a = int(len(self.avg_tf) * self.args.beta)

        
        for i in range(a):
            rank_list.append(self.avg_tf[i])
        self.avg_tf = dict(rank_list)

        
        for key in list(self.motif_count.keys()):
            if key not in self.avg_tf:
                self.motif_count.pop(key)


        for motif_graph in self.motif_dataset:
            for key in list(motif_graph.motif_dict.keys()):
                if key not in self.avg_tf:
                    motif_graph.motif_dict.pop(key)

        for key in self.motif_count.keys():
            self.prototype[key] = []
        for motif_graph in self.motif_dataset:
            self.motifset_dict.append(motif_graph.motif_dict)
       

        
        
        
        


    def prototype_update(self):
       






        for idx, motif_graph in enumerate(self.motif_dataset):
            motif_graph.motif_dict = self.motifset_dict[idx]
        



        
            


        for motif_graph in self.motif_dataset:
            motif_graph.graph.to(self.args.device)
            _, x1 = self.model(motif_graph.graph)

            x1 = x1.data

            for key in motif_graph.motif_dict.keys():
 
                self.prototype[key].append(x1)
                


        
        
        for key in self.prototype.keys():
            if len(self.prototype[key]) == 1:
                self.prototype[key] = self.prototype[key][0].squeeze()
            elif len(self.prototype[key]) > 1:
                c = self.prototype[key][0]
                for i in range(1, len(self.prototype[key])):
                    
                    
                    c = torch.cat((c, self.prototype[key][i]), dim=0)
                self.prototype[key] = c

                
                self.prototype[key] = torch.mean(self.prototype[key], dim=0).data

        
        print('done')
        
    def clear_prototype(self):


        self.prototype_code = {}
        self.motif_code = {} 
        self.prototype_code = {}
    
    def download_code(self, server):
        for key in self.prototype.keys():
            self.prototype_code[key] = server.global_prototype_code[key]
        for key, value in self.prototype_code.items():
            self.code_motif[value] = key
        


    def prototype_train(self, server):

        for key in self.motif_count.keys():
            self.prototype[key] = []
        
        for i, motif_graph in enumerate(self.motif_dataset):
            motif_graph.motif_dict = self.motifset_dict[i]
       
        
        
        for motif_graph in self.motif_dataset:
            
            motif_list = list(motif_graph.motif_dict.keys())
            motif_code = []
            for item in motif_list:
                motif_code.append(self.prototype_code[item])
            motif_graph.motif_dict = tuple(motif_code)

        for motif_graph in self.motif_dataset:
            if motif_graph.motif_dict in self.motif_code.keys():
                motif_graph.motif_dict = self.motif_code[motif_graph.motif_dict]
            else:
                self.motif_code[motif_graph.motif_dict] = len(self.motif_code.keys())
                motif_graph.motif_dict = self.motif_code[motif_graph.motif_dict]



            

        dataset = MotifDataset(self.motif_dataset)

        trainloader = DataLoader(dataset, batch_size=128, shuffle = True)

        self.model.train()
        for batch in trainloader:
            proto = {}
            
            self.optimizer.zero_grad()
            graph, motifs = batch[0].to(self.args.device), batch[1].to(self.args.device)
            pred, protos = self.model(graph)
            

            label = graph.y
            loss1 = self.model.loss(pred, label)
            loss2 = 0.
            loss_mse = nn.MSELoss()
            for i, motif in enumerate(motifs):
                motif_idx = motif.item()
                motif_tuple = list(self.motif_code.keys())[motif_idx]
                for idx in motif_tuple:
                    if idx not in proto.keys():
                        proto[idx] = []
                        proto[idx].append(protos[i])
                        self.prototype[self.code_motif[idx]].append(protos[i].unsqueeze(dim=0))
                    else:
                        proto[idx].append(protos[i])
                        self.prototype[self.code_motif[idx]].append(protos[i].unsqueeze(dim=0))
                        
            
            


            for key in proto.keys():
                if len(proto[key]) == 1:
                    proto[key] = proto[key][0].squeeze()
                elif len(proto[key]) > 1:
                    c = proto[key][0]
                    for i in range(1, len(proto[key])):
                        
                        c = torch.cat((c, proto[key][i]), dim=0)
                    proto[key] = c

                    proto[key] = torch.mean(proto[key], dim=0)




        









            

                    

            proto_global = {}

            for key in proto.keys():
                proto_global[key] = server.global_prototype[server.code[key]].data
            
            for key in proto.keys():

                loss2 += loss_mse(proto[key], proto_global[key]) / len(proto.keys())

            loss = loss1 + loss2 * self.args.lamb
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            proto = {}


        for key in self.prototype.keys():
            if len(self.prototype[key]) == 1:
                
                self.prototype[key] = self.prototype[key][0].squeeze().data
                
            elif len(self.prototype[key]) > 1:
                c = self.prototype[key][0]
                
                for i in range(1, len(self.prototype[key])):
                    
                    
                    c = torch.cat((c, self.prototype[key][i]), dim=0)
                self.prototype[key] = c

                
                self.prototype[key] = torch.mean(self.prototype[key], dim=0).data





        print('finish')
    def cosine_similar(self, server):
        
        similarity = torch.zeros(len(self.prototype.keys())).to(self.args.device)
        for idx, key in enumerate(self.prototype.keys()):
            sim = F.cosine_similarity(self.prototype[key], server.global_prototype[key], 0, 1e-10)
            self.simi[key] = sim
            similarity[idx] = sim
        reput = torch.mean(similarity)

        if len(self.rs) == 0:
            for key in self.simi.keys():
                self.rs[key] = 0.05 * self.simi[key]
        else:
            for key in self.simi.keys():
                self.rs[key] = 0.95 * self.rs[key] + 0.05 * self.simi[key]
        for key in self.rs.keys():
            self.rs[key] = torch.clamp(self.rs[key], min=1e-3)
        
        return reput


    @staticmethod
    def shift_right(l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')




    @staticmethod
    def find_ring_weights(ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            if 'edge_attr' in g.get_edge_data(ring[i], ring[i+1]):
                weight = g.get_edge_data(ring[i], ring[i+1])['edge_attr']
                weight = weight.index(max(weight))
                
            else:
                weight = 1
            weight_list.append(weight)
        if 'edge_attr' in g.get_edge_data(ring[-1], ring[0]):
            weight = g.get_edge_data(ring[-1], ring[0])['edge_attr']
            weight = weight.index(max(weight))
        else:
            weight = 1   
        
        weight_list.append(weight)
        return weight_list
    
    
    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update(self, local_epoch):
        """ For GCFL """
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self):
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)

    def local_train_prox(self, local_epoch, mu):
        """ For FedProx """
        train_stats = train_gc_prox(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device,
                               self.gconvNames, self.W, mu, self.W_old)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate_prox(self, mu):
        return eval_gc_prox(self.model, self.dataLoader['test'], self.args.device, self.gconvNames, mu, self.W_old)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm

def train_gc(model, dataloaders, optimizer, local_epoch, device):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}

def eval_gc(model, test_loader, device):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_gc_prox(model, dataloaders, optimizer, local_epoch, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    convGradsNorm = []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_gc(model, val_loader, device)
        loss_tt, acc_tt = eval_gc(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test, 'convGradsNorm': convGradsNorm}

def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs