import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.linalg import norm
from copy import deepcopy
from utils import *
import matplotlib.pyplot as plt
import scipy.stats



def run_reput(clients, server, communication_rounds, local_epoch, samp=None, frac=1.0):
    rs = torch.zeros(len(clients))
    proto = []
    for i in range(len(rs)):
        rs[i] = 1 / len(rs)
    for client in clients:
        client.motif_construction()
        print(len(client.prototype.keys()))
        proto.append(len(client.prototype.keys()))
    

    proto = torch.tensor(proto, dtype = torch.long)
    proto = proto / torch.max(proto)

    selected_clients = clients
    weight = {}
    for c_round in range(1, communication_rounds+1):

        for i in range(len(selected_clients)):
            selected_clients[i].reput = rs[i]
        weight = {}
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        

        # global aggregate update    
        if c_round == 1:
            for client in selected_clients:
                
                client.prototype_update()
            server.aggregate_prototype(selected_clients)
            # for client in selected_clients:
            #     for motif in client.prototype.keys():
            #         client.rs[motif] = 0.05 * F.cosine_similarity(client.prototype[motif], server.global_prototype[motif], 0, 1e-10)
            #         client.rs[motif] = torch.clamp(client.rs[motif], min=1e-3)

            #print(len(server.global_prototype.keys()))       
        else:
            server.reput3_prototype(selected_clients)
            
            
            print(len(server.global_prototype.keys()))
        # print(len(server.global_prototype.keys())) 
        # update the local model and calculate the local gradient
        gradients = []
        for client in selected_clients:
            old = deepcopy(client.model)
            client.download_code(server)
            client.prototype_train(server)
            new = deepcopy(client.model)
            local_gradient = [(new_param.data - old_param.data) for old_param, new_param in zip(old.parameters(), new.parameters())]
            gradient = []
            for i in range(2, 14):
            # for i in range(len(local_gradient)):
                gradient.append(local_gradient[i])
            
            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7

            gradient = unflatten(torch.div(flattened, norm_value), gradient)
            gradients.append(gradient)
        global_gradient = [torch.zeros(param.shape).to(server.device) for param in server.model.parameters()]
        for gradient, weight in zip(gradients, rs):
            if weight < 0:
                continue
            else:
                add_gradient_updates(global_gradient, gradient, weight)




        s = torch.sum(F.relu(rs)).item()
        for gradient in global_gradient:
            gradient = torch.div(gradient, s)



        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flatten(global_gradient), 0, 1e-10) for gradient in gradients], device=server.device)
        for i, client in enumerate(clients):
            rs[i] = 0.95 * rs[i] + 0.05 * phis[i]
            # ablation
            rs[i] *= proto[i]
        
        #rs = torch.div(rs, rs.sum())
        for i, client in enumerate(clients):
            rs[i] = len(client.dataLoader['train'])
            # ablation
        rs /= rs.sum()
        

        # money payoff
        for i, client in enumerate(selected_clients):
            if c_round == 1:
                client.payoff += rs[i]
            else:
                k = np.mean(np.array(client.reputation))
                client.payoff += rs[i] + torch.max(torch.tensor([rs[i] - k, 0]))
            client.reputation.append(rs[i])




        
        


        # distribute model parameters
        
        q_ratios = torch.tanh(0.5 * rs)
        q_ratios /= torch.max(q_ratios)

        for i in range(len(selected_clients)):
            
            reward_gradient = mask_grad_update_by_order(global_gradient, mask_percentile=q_ratios[i], mode='layer')
            # ablation
            reward_gradient = global_gradient
            for j, k in enumerate(server.W.keys()):
                selected_clients[i].W[k] = selected_clients[i].W[k] + reward_gradient[j]
        

        print('finish2')


        for client in selected_clients:
            client.clear_prototype()
        server.clear_prototype()

    
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    for client in clients:
        print(client.payoff)
    fs = frame.style.apply(highlight_max).data
    print(fs)


    return frame








        
        

        

        


    

    
    
        

    

    
