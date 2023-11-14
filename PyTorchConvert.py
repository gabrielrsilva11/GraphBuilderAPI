from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
import torch
from sklearn import preprocessing
from torch_geometric.data import HeteroData
import pandas as pd

def fetch_graph(random_list):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    graph = ntx.fetch_graph(random_list)
    number_of_words = 0
    le = preprocessing.LabelEncoder()
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
            graph.add((URIRef(s), URIRef("http://ieeta-bit.pt/wikiner#wikinerEntity"), Literal("No")))

    return graph

def fetch_graph_info(graph, nodes_uri, edges_uri):
    node_df = pd.DataFrame()
    edge_df = pd.DataFrame()
    index_list = []
    for s, p, o in graph.triples((None, RDF.type, URIRef(nodes_uri))):
        print(s, p, o)
        node_df.index = s.__str__()
        index_list.append(s.__str__())
        for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
            if p2.__str__() in edges_uri:
                print("YO I AM EDGE")
                print(s2, p2, o2)
            else:
                node_feat_split = p2.__str__().split("#")
                feat_name = node_feat_split[-1]
                print("I AM NODE FEATURE")
                print(feat_name)
                node_df[feat_name] = [o2.__str__()]
    return node_df


words_uri = "http://ieeta-bit.pt/wikiner#Word"
sentence_uri = "http://ieeta-bit.pt/wikiner#Sentence"

depgraph_uri = "http://ieeta-bit.pt/wikiner#depGraph"
head_uri = "http://ieeta-bit.pt/wikiner#head"
previous_uri = "http://ieeta-bit.pt/wikiner#previousWord"
fromsentence_uri = "http://ieeta-bit.pt/wikiner#fromSentence"
fromtext_uri = "http://ieeta-bit.pt/wikiner#fromText"

graph = fetch_graph([1])
nodes = fetch_graph_info(graph, words_uri, [depgraph_uri, head_uri, previous_uri, fromsentence_uri])
print(nodes)


# data = HeteroData()
# --- NODES ---
# data['word']
# data['sentence']

# --- EDGES ---
# --- word edges ---
# data['word', 'depgraph', 'word']
# data['word', 'head', 'word']
# data['word', 'previousWord', 'word]
# data['word', 'fromSentence', 'sentence']

# --- sentence edges ---
# data['sentence', '']