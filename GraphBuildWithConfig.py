import yaml
from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal


def fetch_graph(ids_to_fetch, config):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    edges = {}
    edges_list = []
    for idx in ids_to_fetch:
        graph = ntx.fetch_graph([idx])
        #Go through all the node types and build the graph
        for layer in config_data['nodes']:
            print(layer['name'])
            for s, p, o in graph.triples((None, RDF.type, URIRef(layer['uri']))):
                entity = ''
                print(s, p, o)
                for i in range(0, len(layer['edges'])):
                    print(layer['edges'][i])
                    print(layer['edges_uri'][i])
                    for s2, p2, o2 in graph.triples((URIRef(s), URIRef(layer['edges_uri'][i]), None)):
                        print(s2, p2, o2)
                        #CRIAR UM DICIONARIO COM AS EDGES


def fetch_graph_info(graph, nodes_uri, edges_uri, edges_names, index_list, nodes_list, edges_list):
    for s, p, o in graph.triples((None, RDF.type, URIRef(nodes_uri))):
        node_features = {}
        edge_features = {}
        # Index to build the node/edges dataframe.
        # In the edges this index will be the "source" node
        index_list.append(s.__str__())
        for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
            # Add the edges information
            if p2.__str__() in edges_uri:
                list_index = edges_uri.index(p2.__str__())
                print(list_index)
                edge_features[edges_names[list_index]] = [s2.__str__(), o2.__str__()]
            # Add the node features
            else:
                node_feat_split = p2.__str__().split("#")
                feat_name = node_feat_split[-1]
                node_features[feat_name] = o2.__str__()
        edges_list.append(edge_features)
        nodes_list.append(node_features)
    return nodes_list, edges_list, index_list

config_file = open('conf.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)
print(config_data)

for layer in config_data['nodes']:
    print(layer)

for layer in config_data['target']:
    print(layer['name'])
    print(layer['uri'])

fetch_graph([1,2,3], config_data)