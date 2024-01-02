import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw


class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []
        self.vocab = {}
        self.device = device
        
        self.whole_node_count = {}
        self.num_client = {}
        self.global_prototype = {}
        self.global_prototype_code = {}
        self.weight = {}
        self.code_prototype = {}
        self.code = {}





    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]



    def aggregate_prototype(self, clients):
        for client in clients:
            for key in client.motif_count.keys():
                if key not in self.vocab.keys():
                    self.vocab[key] = client.motif_count[key]
                    self.num_client[key] = 1
                else:
                    self.vocab[key] += client.motif_count[key]
                    self.num_client[key] += 1
        for client in clients:
            for key in client.motif_count.keys():
                if key not in self.global_prototype.keys():
                    self.global_prototype[key] = client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]
                    
                else:
                    
                    self.global_prototype[key] += client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]
                    
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data

        for i, motif in enumerate(list(self.global_prototype.keys())):
            self.global_prototype_code[motif] = i
            self.code[i] = motif





    def aggregate_code(self, clients):
        for client in clients:
            for key in client.motif_count.keys():
                if key not in self.vocab.keys():
                    self.vocab[key] = client.motif_count[key]
                    self.num_client[key] = 1
                else:
                    self.vocab[key] += client.motif_count[key]
                    self.num_client[key] += 1
        for client in clients:
            for key in client.motif_count.keys():
                if key not in self.global_prototype.keys():
                    self.global_prototype[key] = client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]
                else:
                    self.global_prototype[key] += client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data

        for i, motif in enumerate(list(self.global_prototype.keys())):
            self.global_prototype_code[motif] = i
            self.code[i] = motif







    


    def reput_aggregate_prototype(self, rs, clients):
        
        for i, client in enumerate(clients):
            
            for key in client.motif_count.keys():
                #weight = 0
                if key not in self.global_prototype.keys():
                    self.global_prototype[key] = rs[i] * client.prototype[key]
                    
                    self.weight[key] = rs[i]
                else:
                    self.global_prototype[key] += rs[i] * client.prototype[key]
                    self.weight[key] += rs[i]
        for key in self.global_prototype.keys():
            self.global_prototype[key] /= self.weight[key]
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data
        for i, motif in enumerate(list(self.global_prototype.keys())):
            self.global_prototype_code[motif] = i
            self.code[i] = motif

        
        



    def reput_aggregate_prototype2(self, rs, clients):
        
        for i, client in enumerate(clients):
            
            for key in client.motif_count.keys():
                #weight = 0
                if key not in self.global_prototype.keys():

                    self.global_prototype[key] = client.rs[key] * client.prototype[key]
                    self.weight[key] = client.rs[key]
                else:
                    self.global_prototype[key] += client.rs[key] * client.prototype[key]
                    self.weight[key] += client.rs[key]
        for key in self.global_prototype.keys():
            self.global_prototype[key] /= self.weight[key]
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data
        for i, motif in enumerate(list(self.global_prototype.keys())):
            self.global_prototype_code[motif] = i
            self.code[i] = motif
        
        
        
    def update_reput(self, clients):
        for key in self.global_prototype.keys():
            weight = 0
            for client in clients:
                if key in client.rs.keys():
                    weight += client.rs[key]
            for client in clients:
                if key in client.rs.keys():
                    client.rs[key] /= weight
    
    def reput3_prototype(self, clients):
        for client in clients:
            for key in client.motif_count.keys():
                if key not in self.vocab.keys():
                    self.vocab[key] = client.motif_count[key]
                    self.num_client[key] = 1
                else:
                    self.vocab[key] += client.motif_count[key]
                    self.num_client[key] += 1
        for i, client in enumerate(clients):
            for key in client.motif_count.keys():
                if key not in self.global_prototype.keys():
                    self.global_prototype[key] = client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]
                else:
                    self.global_prototype[key] += client.motif_count[key] / self.vocab[key] * client.prototype[key] / self.num_client[key]   
        for key in self.global_prototype.keys():
            self.global_prototype[key] = self.global_prototype[key].data
        for i, motif in enumerate(list(self.global_prototype.keys())):
            self.global_prototype_code[motif] = i
            self.code[i] = motif
            

        
                    
                


        



            
        


        






    def clear_prototype(self):
        self.vocab = {}
        
        self.whole_node_count = {}
        self.num_client = {}
        self.global_prototype = {}
        self.global_prototype_code = {}
        self.code_prototype = {}

        
        self.weight = {}

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()

def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp