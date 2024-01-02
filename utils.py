import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
import math


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    # graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df



def flatten(grad_update):
	return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
		flattened = flattened[n_params:]

	return grad_update
def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update




def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):
    if mode == 'layer':
        grad_update = copy.deepcopy(grad_update)
        mask_percentile = max(0, mask_percentile)
        for i, layer in enumerate(grad_update):
            
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)
            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
                
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update

    elif mode == 'all':
        all_update_mod = torch.cat([update.data.view(-1).abs() for update in grad_update])
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)
            return mask_grad_update_by_magnitude(grad_update, topk[-1])
        


    # elif mode == 'all':
	# 	# mask all but the largest <mask_order> updates (by magnitude) to zero
	#     all_update_mod = torch.cat([update.data.view(-1).abs() for update in grad_update])
									
    #     if not mask_order and mask_percentile is not None:
	# 	    mask_order = int(len(all_update_mod) * mask_percentile)
		
	#     if mask_order == 0:
	# 	    return mask_grad_update_by_magnitude(grad_update, float('inf'))
	#     else:
	# 	    topk, indices = torch.topk(all_update_mod, mask_order)
	# 	    return mask_grad_update_by_magnitude(grad_update, topk[-1])

    # elif mode == 'layer': # layer wise largest-values criterion
    #     grad_update = copy.deepcopy(grad_update)
    #     print(grad_update)

    #     mask_percentile = max(0, mask_percentile)
    #     for i, layer in enumerate(grad_update):
    #         layer_mod = layer.data.view(-1).abs()
    #         if mask_percentile is not None:
    #             mask_order = math.ceil(len(layer_mod) * mask_percentile)

    #         if mask_order == 0:
    #             grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
    #         else:
    #             topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))

                                                                                                                                                                                            
    #             grad_update[i].data[layer.data.abs() < topk[-1]] = 0
    #     return grad_update

    
def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
	assert len(grad_update_1) == len(
		grad_update_2), "Lengths of the two grad_updates not equal"
	
	for param_1, param_2 in zip(grad_update_1, grad_update_2):
		param_1.data += param_2.data * weight