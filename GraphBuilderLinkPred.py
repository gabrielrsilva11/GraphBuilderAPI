from tqdm import tqdm
from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import random

def fetch_graph(edges, nodes, indexes, ids_to_fetch, config_data, test):
    print("--- LOADING DATASET ---")
    base_uri = config_data['connection'][0]['base_uri']
    graph_name = config_data['connection'][0]['graph_name']
    conection_string = config_data['connection'][0]['connection_uri']
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    processed = False
    for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
        #for idx in ids_to_fetch:
        idx = ids_to_fetch[i]
        graph = ntx.fetch_graph([idx])
        # entity = False
        # if config_data['target'][0]['balancing']:
        #     for s3, p3, o3 in graph.triples((None, URIRef(config_data['target'][0]['uri'][0]), None)):
        #         if o3.__str__():
        #             entity = True
        # chance = random.random()
        # if chance < 0.1:
        for layer in config_data['nodes']:
            if layer['name'] not in indexes:
                indexes[layer['name']] = []
            if layer['name'] not in nodes:
                nodes[layer['name']] = []
            if layer['name'] != "word" and layer['name'] != 'sentence' and not processed:
                graph = ntx.fetch_graph_with_query(graph, layer['name'])
                processed=True

            for s, p, o in graph.triples((None, RDF.type, URIRef(layer['uri']))):
                #node_type = s.__str__().split("#")[-1].split("_")[0]
                node_features = {}
                num_edges = 0
                for s2, p2, o2 in graph.triples((URIRef(s), None, None)):
                    uri_type = p2.__str__().split("#")[-1]
                    #An edge to add
                    if uri_type in layer['edges']:
                        if uri_type not in edges:
                            edges[uri_type] = []
                        edges[uri_type].append([s2.__str__(), o2.__str__()])
                        num_edges += 1
                    #Node Attribute
                    else:
                        node_features[uri_type] = o2.__str__()
                if "softwareMention" not in edges:
                    print("NO MENTION")
                if s.__str__() not in indexes[layer['name']]:
                    indexes[layer['name']].append(s.__str__())

                if node_features:
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


def build_node_relationships(uniqueIds, nodeList, source_name, target_name, balancing):
    original_df = pd.DataFrame(data=nodeList, columns=["source", "target"])
    if target_name in uniqueIds:
        source_df = pd.merge(original_df['source'], uniqueIds[source_name],
                             left_on='source', right_on='originalId', how='left')
        target_df = pd.merge(original_df['target'], uniqueIds[target_name],
                             left_on='target', right_on='originalId', how='left')
    else:
        source_df = pd.merge(original_df['source'], uniqueIds[source_name],
                             left_on='source', right_on='originalId', how='left')
        unique_targets = pd.DataFrame(data={
            'originalId': original_df['target'].unique(),
            'mappedId': pd.RangeIndex(len(original_df['target'].unique()))
        })
        target_df = pd.merge(original_df['target'], unique_targets,
                             left_on='target', right_on='originalId', how='left')

    #target_df = target_df.dropna()
    target_df['mappedId'] = target_df['mappedId'].astype(int)
    index_targets = target_df.index
    source_df = source_df.iloc[index_targets]

    source = torch.from_numpy(source_df['mappedId'].values)
    target = torch.from_numpy((target_df['mappedId'].values))
    node_tensor = torch.stack([source, target], dim=0)
    return node_tensor

def build_targets(nodes_df, config_data, test=False, targets_test=None):
    nodes_df[config_data['target'][0]['name'][0]].fillna("No", inplace=True)

    #Replace names with their first name
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "Application_Creation", "Application")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "Application_Mention", "Application")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "Application_Usage", "Application")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "Application_Deposition", "Application")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "PlugIn_Creation", "PlugIn")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "PlugIn_Mention", "PlugIn")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "PlugIn_Usage", "PlugIn")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "PlugIn_Deposition", "PlugIn")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "ProgrammingEnvironment_Usage", "ProgrammingEnvironment")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "ProgrammingEnvironment_Mention", "ProgrammingEnvironment")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "SoftwareCoreference_Deposition", "SoftwareCoreference")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "OperatingSystem_Usage", "OperatingSystem")
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "OperatingSystem_Mention", "OperatingSystem")

    print(nodes_df[config_data['target'][0]['name'][0]].value_counts())
    print(nodes_df[config_data['target'][0]['name'][0]])

    if test:
        unique_targets_df = targets_test
    else:
        unique_targets_id = nodes_df[config_data['target'][0]['name'][0]].unique()
        unique_targets_df = pd.DataFrame(data={
            'originalId': unique_targets_id,
            'mappedId': pd.RangeIndex(len(unique_targets_id))
        })
    targets_df = pd.merge(nodes_df[config_data['target'][0]['name'][0]], unique_targets_df,
                          left_on=config_data['target'][0]['name'][0], right_on='originalId', how='left')
    targets = torch.from_numpy(targets_df['mappedId'].values)
    if test:
        return targets, unique_targets_df, len(nodes_df[config_data['target'][0]['name'][0]].unique())

    return targets, unique_targets_df, len(unique_targets_df)

def build_graph(nodes, edges, mapped_ids, config_data, graph_data , test= False, test_targets=None):
    for layer in config_data['nodes']:
        column_list = []

        nodes_df = pd.DataFrame.from_records(nodes[layer['name']])
        nodes_df.index = mapped_ids[layer['name']]['mappedId']
        for attributes in layer['attributes']:
            if attributes in nodes_df.columns:
                column_list.append(attributes)
                nodes_df[attributes] = nodes_df[attributes].astype('category').cat.codes
            else:
                print("---- WARNING ----")
                print(attributes, " is present in the config file but was not found in the data.")
                print("-----------------")
        # if layer['name'] == config_data['target'][0]['node']:
        #     nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(r'\n','', regex=True)
            # if test:
            #     targets, unique_targets, size_targets = build_targets(nodes_df, config_data, test=test, targets_test = test_targets)
            # else:
            #     targets, unique_targets, size_targets = build_targets(nodes_df, config_data)
            # graph_data[layer['name']].y = targets
            # graph_data.num_classes = size_targets
        nodes_appropriate = nodes_df[column_list]
        nodes_tensor = torch.from_numpy(nodes_appropriate.values).to(torch.float)
        graph_data[layer['name']].x = nodes_tensor
        for edge_idx in range(0, len(layer['edges'])):
            graph_data[layer['edges_source'][edge_idx], layer['edges'][edge_idx], layer['edges_target'][edge_idx]].edge_index = build_node_relationships(mapped_ids, edges[layer['edges'][edge_idx]], layer['edges_source'][edge_idx], layer['edges_target'][edge_idx], config_data['target'][0]['balancing'])
    return graph_data#, unique_targets, size_targets

def get_graph(list_to_get, config_data, test=False, targets_test = None):
    graph_data = HeteroData()
    edges = {}
    nodes = {}
    indexes = {}
    edges, nodes, indexes = fetch_graph(edges, nodes, indexes, list_to_get, config_data, test)
    #In our case we have to pop the last element of the list while testing.
    #This is due to a sentence referencing a non-existant next sentence.
    #This will not happen with complete texts and should be removed.
    print(edges['nextSentence'])
    edges['nextSentence'].pop()
    print(edges['nextSentence'])
    mapped_uris = map_uri_to_index(indexes, config_data)

    if test:
        graph_data, unique_targets, size_targets = build_graph(nodes, edges, mapped_uris, config_data, graph_data, test=test, test_targets=targets_test)
    else:
        graph_data = build_graph(nodes, edges, mapped_uris, config_data, graph_data)
    print("--- DATASET LOADED AND TRANSFORMED ---")
    return graph_data, nodes, mapped_uris