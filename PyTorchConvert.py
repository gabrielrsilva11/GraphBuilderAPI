from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
import torch
from sklearn import preprocessing
from torch_geometric.utils.convert import from_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OGB_MAG


def load_dataset_wikiner(start, stop, random_list):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(random_list)
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

    g_ntx = rdflib_to_networkx_digraph(graph)
    pyg = from_networkx(g_ntx)
    print(pyg)

def load_wikiner_dataset_pytorch(random_list):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    wikinerentity_uri = base_uri + "wikinerEntity"
    head_uri = base_uri + "head"
    depgraph_uri = base_uri + "depGraph"
    nextsentence_uri = base_uri + "nextSentence"
    previousword_uri = base_uri + "previousWord"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(random_list)
    label_encoder_list = []
    features_list = []
    targets_list = []
    nodes_list = []
    adj = []
    for s, p, o in graph.triples((None, None, None)):
        #print(s, p, o)
        #print("\n")

        #Get all the nodes
        if s not in nodes_list:
            nodes_list.append(s.__str__())

        #Get all the unique nodes for a label encoder later.
        if s not in label_encoder_list:
            label_encoder_list.append(s.__str__())
        if p not in label_encoder_list:
            label_encoder_list.append(p.__str__())
        if o not in label_encoder_list:
            label_encoder_list.append(o.__str__())

        if p.__str__() == wikinerentity_uri:
            targets_list.append([s.__str__(), o.__str__()])
        elif p.__str__() == head_uri or p.__str__() == nextsentence_uri or p.__str__() == previousword_uri or p.__str__() == depgraph_uri:
            adj.append([s.__str__(), o.__str__()])
        else:
            features_list.append(o.__str__())

    le = preprocessing.LabelEncoder()
    enconded_list = le.fit_transform(label_encoder_list)
    nodes = le.transform(nodes_list)

    targets = []
    for target in targets_list:
        transformed_target = le.transform(target)
        targets.append([transformed_target[0], transformed_target[1]])

    edge_index_list = []
    for edge in adj:
        edges_transformed = le.transform(edge)
        edge_index_list.append([edges_transformed[0], edges_transformed[1]])
    print(targets)
    print(edge_index_list)


#load_wikiner_dataset_pytorch([1,3,5,7])
dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
data = dataset[0]
print(data)