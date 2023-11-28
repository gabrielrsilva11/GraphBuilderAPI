import yaml
from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def fetch_graph(ids_to_fetch, config_data):
    print("--- LOADING DATASET ---")
    base_uri = "http://ieeta-bit.pt/wikiner#"
    graph_name = "WikiNER"
    conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    edges = {}
    nodes = {}
    indexes = {}
    for idx in ids_to_fetch:
        graph = ntx.fetch_graph([idx])
        for layer in config_data['nodes']:
            #Go through all the desirable nodes
            if layer['name'] not in indexes:
                indexes[layer['name']] = []
            if layer['name'] not in nodes:
                nodes[layer['name']] = []
            for s, p, o in graph.triples((None, RDF.type, URIRef(layer['uri']))):
                if s.__str__() not in indexes[layer['name']]:
                    indexes[layer['name']].append(s.__str__())
                node_type = s.__str__().split("#")[-1].split("_")[0]
                node_features = {}
                for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
                    uri_type = p2.__str__().split("#")[-1]
                    #An edge to add
                    if uri_type in layer['edges']:
                        if uri_type not in edges:
                            edges[uri_type] = []
                        edges[uri_type].append([s2.__str__(), o2.__str__()])
                    #Node Attribute
                    else:
                        node_features[uri_type] = o2.__str__()
                if config_data['target'][0]['name'][0] not in node_features and node_type == config_data['target'][0]['node']:
                    node_features["wikinerEntity"] = "No"
                nodes[layer['name']].append(node_features)
    return edges, nodes, indexes


def map_uri_to_index(indexes, config_data):
    unique_ids = {}
    for layer in config_data['nodes']:
        unique_ids_df = pd.DataFrame(data={
            'originalId': indexes[layer['name']],
            'mappedId': pd.RangeIndex(len(indexes[layer['name']]))
        })
        unique_ids[layer['name']] = unique_ids_df
    return unique_ids


def build_node_relationships(uniqueIds, nodeList, source_name, target_name):
    original_df = pd.DataFrame(data=nodeList, columns=["source", "target"])
    source_df = pd.merge(original_df['source'], uniqueIds[source_name],
                         left_on='source', right_on='originalId', how='left')
    source = torch.from_numpy(source_df['mappedId'].values)
    target_df = pd.merge(original_df['target'], uniqueIds[target_name],
                         left_on='target', right_on='originalId', how='left')
    target = torch.from_numpy((target_df['mappedId'].values))
    node_tensor = torch.stack([source, target], dim=0)
    return node_tensor

def build_targets(nodes_df, config_data):
    unique_targets_id = nodes_df[config_data['target'][0]['name'][0]].unique()
    unique_targets_df = pd.DataFrame(data={
        'originalId': unique_targets_id,
        'mappedId': pd.RangeIndex(len(unique_targets_id))
    })
    targets_df = pd.merge(nodes_df['wikinerEntity'], unique_targets_df,
                          left_on='wikinerEntity', right_on='originalId', how='left')
    targets = torch.from_numpy(targets_df['mappedId'].values)
    return targets, len(unique_targets_df)

def build_graph(nodes, edges, mapped_ids, config_data):
    graph_data = HeteroData()
    for layer in config_data['nodes']:
        column_list = []
        nodes_df = pd.DataFrame.from_records(nodes[layer['name']])
        nodes_df.index = mapped_ids[layer['name']]['mappedId']
        for attributes in layer['attributes']:
            column_list.append(attributes)
            nodes_df[attributes] = nodes_df[attributes].astype('category').cat.codes
        if layer['name'] == config_data['target'][0]['node']:
            targets, size_targets = build_targets(nodes_df, config_data)
            graph_data[layer['name']].y = targets
            graph_data.num_classes = size_targets
        nodes_appropriate = nodes_df[column_list]
        nodes_tensor = torch.from_numpy(nodes_appropriate.values).to(torch.float)
        graph_data[layer['name']].x = nodes_tensor
        for edge_idx in range(0, len(layer['edges'])):
            print(layer['edges'][edge_idx])
            graph_data[layer['edges_source'][edge_idx], layer['edges'][edge_idx], layer['edges_target'][edge_idx]].edge_index = build_node_relationships(mapped_ids, edges[layer['edges'][edge_idx]], layer['edges_source'][edge_idx], layer['edges_target'][edge_idx])

    return graph_data


def get_graph(list_to_get, config_data):
    edges, nodes, indexes = fetch_graph(list_to_get, config_data)
    mapped_uris = map_uri_to_index(indexes, config_data)
    #In our case we have to pop the last element of the list while testing.
    #This is due to a sentence referencing a non-existant next sentence.
    #This will not happen with complete texts and should be removed.
    edges['nextSentence'].pop()
    graph = build_graph(nodes, edges, mapped_uris, config_data)
    return graph
