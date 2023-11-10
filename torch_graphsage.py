import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import os.path as osp
import pandas as pd
import numpy as np
import collections
from pandas.core.common import flatten
# importing obg datatset
from pandas.core.common import flatten
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from scipy.special import softmax
import umap
from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit


sns.set(rc={'figure.figsize':(16.7,8.27)})
sns.set_theme(style="ticks")

def load_testing_dataset(start, stop):
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(start, stop)
    add = 0
    number_of_words = 0
    for s, p, o in graph.triples((None, RDF.type, URIRef("http://ieeta-bit.pt/wikiner#Word"))):
        number_of_words += 1
        entity = ''
        for s2, p2, o2 in graph.triples((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), None)):
            entity = o2
        if entity:
            entity_no_break = entity.split("\n")
            entity_correct = entity_no_break[0].split("-")
            graph.remove((s2, p2, o2))
            graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal(entity_correct[1])))
        else:
            add += 1
            if add == 20:
                graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal("No")))
                add = 0

    triples = pd.DataFrame(
        ((s.n3(), str(p), o.n3()) for s, p, o in graph),
        columns=["source", "label", "target"],
    )
    print(triples)
    all_nodes = pd.concat([triples.source, triples.target])
    nodes = pd.DataFrame(index=pd.unique(all_nodes))
    nodes_onehot_features = pd.get_dummies(nodes.index).set_index(nodes.index)
    x = torch.tensor(nodes_onehot_features.values)

    edges = {
        edge_type: df.drop(columns="label")
        for edge_type, df in triples.groupby("label")
    }
    print(set(triples.label))
    affiliation = edges.pop("http://ieeta-bit.pt/wikiner#wikinerEntity")
    onehot_affiliation = pd.get_dummies(affiliation.set_index("source")["target"])
    y = torch.tensor(onehot_affiliation.values)
    data = Data(x=x, edge_index=y)
    print(data)
    #graph = rdflib_to_networkx_graph(graph, edge_attrs=list(set(triples.label)))
    return data


graph_torch = load_testing_dataset(1, 10)
#graph_torch = from_networkx(graph)
print(graph_torch)
transform = RandomLinkSplit(is_undirected=True)
train_data, test_data = transform(graph_torch)
print(train_data)
