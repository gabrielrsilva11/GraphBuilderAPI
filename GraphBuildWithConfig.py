from tqdm import tqdm
from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import random
from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from Helpers import list_conll_subgraph
import pprint
from copy import deepcopy
from statistics import mode

def fetch_graph(edges, nodes, indexes, ids_to_fetch, config_data, test):
    print("--- LOADING DATASET ---")
    if test == 'Test':
        base_uri = config_data['connection'][0]['base_uri_test']
        graph_name = config_data['connection'][0]['graph_name_test']
    else:
        base_uri = config_data['connection'][0]['base_uri']
        graph_name = config_data['connection'][0]['graph_name']

    conection_string = config_data['connection'][0]['connection_uri']
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    embeds = {}
    for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
        idx = ids_to_fetch[i]
        graph = ntx.fetch_graph([idx])
        # ---- CHANGE THE TARGETS TO THE FIRST IN THE CHAIN -----
        # ---- COMMENT FOR REGULAR GRAPH ----
        if test == 'Test' or test == 'Train':
            graph = change_targets(graph, base_uri)
        process = True
        if graph != None:
            for s, p, o in graph.triples((None, URIRef(base_uri + "senttext"), None)):
                #print(s, p, o)
                if len(o.__str__()) < 1000:
                    process = True
                else:
                    process = False
            if process:
                for layer in config_data['nodes']:
                    if layer['name'] not in indexes:
                        indexes[layer['name']] = []
                    if layer['name'] not in nodes:
                        nodes[layer['name']] = []
                    for s, p, o in graph.triples((None, RDF.type, URIRef(layer['uri']))):
                        #node_type = s.__str__().split("#")[-1].split("_")[0]
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

                            # if uri_type == 'senttext':
                            #     sentence = o2.__str__().replace("  ", " ").split(" ")
                            #     if "" in sentence:
                            #         sentence.remove("")
                            #    sentence_id = s2.__str__().split("#")[-1].split("Sentence")[-1]
                                # try:
                                #     embeds[sentence_id] = [sentence, embed_text(sentence, "CLASSIFICATION", "text-multilingual-embedding-preview-0409")]
                                # except:
                                #     print("--- ERROR OCCURED ON GETTING SENTENCE EMBEDDINGS ---")
                                #     print(sentence)
                        node_features['uri'] = s2.__str__().split("#")[-1]
                        if layer['name'] == 'word':
                            if node_features['word'] != '':
                                nodes[layer['name']].append(node_features)
                                add = True
                            else:
                                add = False
                        else:
                            nodes[layer['name']].append(node_features)
                            add = True
                        # nodes[layer['name']].append(node_features)
                        if s.__str__() not in indexes[layer['name']] and add:
                            indexes[layer['name']].append(s.__str__())
    return edges, nodes, indexes

def change_targets(g, base_uri):
    senttext_uri = URIRef(base_uri + "senttext")
    depgraph_uri = URIRef(base_uri + "depGraph")
    id_uri = URIRef(base_uri + "id")
    tags = URIRef(base_uri + "OIE_tags")
    word = URIRef(base_uri + "word")
    tag = URIRef(base_uri + "target")
    graph = False
    for s, p, o in g.triples((None, senttext_uri, None)):
        graph = list_conll_subgraph(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri,
                                    main_uri=base_uri)
    if graph:
        if len(graph) == 3:
            #pprint.pprint(graph)
            # Fetch the words that make up A1, R, A2
            a1 = []
            r = []
            a2 = []
            for s, p, o in g.triples((None, tags, None)):
                if o.__str__() == "A1":
                    for s, p , o in g.triples((s, word, None)):
                        #a1 = a1 + o.__str__() + " "
                        a1.append([s.__str__(), "A1"])
                elif o.__str__() == "A2":
                    for s, p, o in g.triples((s, word, None)):
                        # a2 = a2 + o.__str__() + " "
                        a2.append([s.__str__(), "A2"])
                elif o.__str__() == "R":
                    for s, p, o in g.triples((s, word, None)):
                        #r = r + o.__str__() + " "
                        r.append([s.__str__(), "R"])
                else:
                    g.add((s, URIRef(tag), Literal("No")))
            if a1 and r and a2:
                relations = find_central_relationship(relations=[a1, r, a2], graph=graph)
                g = add_new_target_to_graph(a1, relations, "A1", tag, g)
                g = add_new_target_to_graph(a2, relations, "A2", tag, g)
                g = add_new_target_to_graph(r, relations, "R", tag, g)
            else:
                return None
    return g

def recursive_find(graph_node, word_numbers, current_max_length, list_with_index):
    #print(graph_node)
    if len(word_numbers) == 1:
        list_with_index.append([word_numbers[0], 100])
    else:
        for i in range(0, len(graph_node)):
            if type(graph_node[i]) == list:
                # print(graph_node[i])
                # print(len(graph_node[i]))
                if graph_node[i][0] in word_numbers:
                    #if current_max_length < len(graph_node[i]):
                        #current_max_length = len(graph_node[i])
                        #list_with_index.append([word_numbers.index(graph_node[i][0]), current_max_length])
                    current_max_length = len(graph_node[i])
                    list_with_index.append([graph_node[i][0], current_max_length])
                    for j in range(2, len(graph_node[i])):
                        if len(graph_node[i][j]) > 2:
                            list_with_index = recursive_find(graph_node[i][j], word_numbers, current_max_length,
                                                             list_with_index)
                elif len(graph_node[i]) > 2:
                    for j in range(2, len(graph_node[i])):
                        if len(graph_node[i][j]) > 2:
                            list_with_index = recursive_find(graph_node[i][j], word_numbers, current_max_length, list_with_index)
            else:
                if graph_node in word_numbers:
                    if graph_node[i][2][1].split("#")[-1] == "ROOT":
                        current_idx = word_numbers.index(graph_node[i])
                        #list_with_index.append([current_idx, 10])
                        list_with_index.append([graph_node[i], 100])
                        break
                    elif current_max_length < 1:
                        current_max_length = 1
                        #list_with_index.append(word_numbers.index(graph_node[i]), current_max_length)
                        list_with_index.append(graph_node[i], current_max_length)
    return list_with_index


def get_root_node(graph_as_list: list, w_id: int, id_list: list) -> int:
    _NOT_FOUND_ = -1
    this_w_id, _, *subgraph = graph_as_list
    if this_w_id == w_id:
        return this_w_id
    for sg in subgraph:
        current_root = get_root_node(sg, w_id, id_list)
        if current_root != _NOT_FOUND_:
            return this_w_id if this_w_id in id_list else current_root
    return _NOT_FOUND_


def get_most_frequent_root(graph_as_list: list, word_ids: list) -> int:
    #print(graph_as_list)
    return mode([get_root_node(graph_as_list, int(w_id), word_ids) for w_id in word_ids])


def find_central_relationship(relations, graph):
    first_found = {}
    for relation in range(0, len(relations)):
        word_ids = []
        triple_type = ""
        for numbers in relations[relation]:
            word_ids.append(int(numbers[0].split("_")[-1]))
            triple_type = numbers[1]
        word_index = get_most_frequent_root(graph, word_ids)
        first_found[triple_type] = relations[relation][word_ids.index(word_index)][0]
    return first_found

def add_new_target_to_graph(past_relations, new_relations, rel_type, target_uri, graph):
    relationship = ""
    if rel_type in new_relations:
        relationship = new_relations[rel_type]
    for rel_id in past_relations:
        if rel_id[0] == relationship:
            graph.add((URIRef(rel_id[0]), URIRef(target_uri), Literal(rel_id[1])))
        else:
            graph.add((URIRef(rel_id[0]), URIRef(target_uri), Literal("No")))
    return graph

def map_uri_to_index(indexes, config_data):
    unique_ids = {}
    for layer in config_data['nodes']:
        unique_ids_df = pd.DataFrame(data={
            'originalId': indexes[layer['name']],
            'mappedId': pd.RangeIndex(len(indexes[layer['name']]))
        })
        unique_ids[layer['name']] = unique_ids_df
    return unique_ids

def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "textembedding-gecko@003",
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

def build_node_relationships(uniqueIds, nodeList, source_name, target_name, balancing):
    original_df = pd.DataFrame(data=nodeList, columns=["source", "target"])
    #print(original_df.to_string())
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

    target_df = target_df.dropna()
    target_df['mappedId'] = target_df['mappedId'].astype(int)
    index_targets = target_df.index
    source_df = source_df.iloc[index_targets]

    source = torch.from_numpy(source_df['mappedId'].values)
    #source = source.type(torch.int64)
    target = torch.from_numpy((target_df['mappedId'].values))
    #target = target.type(torch.int64)
    to_remove = torch.isnan(target)
    source = source[to_remove != True]
    target = target[to_remove != True]
    to_remove_source = torch.isnan(source)
    source = source[to_remove_source != True]
    target = target[to_remove_source != True]
    source = source.type(torch.int64)
    target = target.type(torch.int64)
    node_tensor = torch.stack([source, target], dim=0)
    return node_tensor

def build_targets(nodes_df, config_data, test=False, targets_test=None):
    nodes_df[config_data['target'][0]['name'][0]].fillna("0", inplace=True)

    # Replace names with their first name
    nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
        "0", "No")
    # nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
    #     "A1", "A")
    # nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
    #     "A2", "A")
    # nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(
    #     "8", "7")

    print(nodes_df[config_data['target'][0]['name'][0]].value_counts())
    #print(nodes_df[config_data['target'][0]['name'][0]])

    if test == "Test":
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
    if test == "Test":
        return targets, unique_targets_df, len(nodes_df[config_data['target'][0]['name'][0]].unique())

    return targets, unique_targets_df, len(unique_targets_df)

def build_graph(nodes, edges, mapped_ids, config_data, graph_data , test = False, test_targets=None, embedding = True):
    for layer in config_data['nodes']:
        column_list = []
        nodes_df = pd.DataFrame.from_records(nodes[layer['name']])
        nodes_df.index = mapped_ids[layer['name']]['mappedId']
        for attributes in layer['attributes']:
            if attributes in nodes_df.columns:
                column_list.append(attributes)
                if attributes != 'embeddings':
                    nodes_df[attributes] = nodes_df[attributes].astype('category').cat.codes
            else:
                print("---- WARNING ----")
                print(attributes, " is present in the config file but was not found in the data.")
                print("-----------------")
        if test != 'Benchmark':
            if layer['name'] == config_data['target'][0]['node']:
                nodes_df[config_data['target'][0]['name'][0]] = nodes_df[config_data['target'][0]['name'][0]].replace(r'\n','', regex=True)
                if test == "Test":
                    targets, unique_targets, size_targets = build_targets(nodes_df, config_data, test=test, targets_test = test_targets)
                else:
                    targets, unique_targets, size_targets = build_targets(nodes_df, config_data)
                graph_data[layer['name']].y = targets
                graph_data.num_classes = size_targets
        if layer['name'] == 'word':
            if embedding:
                print(column_list)
                column_list.remove('embeddings')

            column_list.remove('word')
            column_list.remove('lemma')
            print(column_list)
            #adicionar aqui uma coluna com o id da palavra

            nodes_appropriate = nodes_df[column_list]
            nodes_tensor = torch.from_numpy(nodes_appropriate.values).to(torch.float)
            if embedding:
                embeds = nodes_df['embeddings'].values
                embeds_tensor = torch.stack(embeds.tolist(), dim=0)
                nodes_tensor = torch.cat((nodes_tensor, embeds_tensor), 1)
        else:
            nodes_appropriate = nodes_df[column_list]
            nodes_tensor = torch.from_numpy(nodes_appropriate.values).to(torch.float)
        graph_data[layer['name']].x = nodes_tensor
        for edge_idx in range(0, len(layer['edges'])):
            try:
                graph_data[layer['edges_source'][edge_idx], layer['edges'][edge_idx], layer['edges_target'][edge_idx]].edge_index = build_node_relationships(mapped_ids, edges[layer['edges'][edge_idx]], layer['edges_source'][edge_idx], layer['edges_target'][edge_idx], config_data['target'][0]['balancing'])
            except:
                print("Error on: ", layer['edges_source'][edge_idx])

    if test == 'Benchmark':
        return graph_data

    return graph_data, unique_targets, size_targets

def check_embed_dict(embeds, words_list):
    list_for_embeds = []
    for word_id in range(0, len(words_list)):
        if words_list[word_id] not in embeds:
            list_for_embeds.append(words_list[word_id])
    return list_for_embeds

def match_embeds_to_words(nodes):
    sentences = []
    sentence = []
    dict_embeds = {}
    #Go through words and build sentences to match indexes that might be missing
    for i in range(0, len(nodes['word'])):
        current_uri = nodes['word'][i]['uri'].split("_")[1:3]
        if i == 0:
            check_uri = current_uri[1]
        elif check_uri != current_uri[1]:
            sentences.append(sentence)
            check_uri = current_uri[1]
            sentence = []
        if nodes['word'][i]['word'] == '':
            nodes['word'][i]['embeddings'] = torch.zeros(768)
        else:
            sentence.append(nodes['word'][i]['word'])
    sentences.append(sentence)
    embeddings = []
    #For each built sentence query google cloud for embeddings
    for i in tqdm(range(len(sentences)), desc="Generating Embeddings: "):
        embed_list = check_embed_dict(dict_embeds, sentences[i])
        complete_embed_list = []
        embeds = []
        if embed_list:
            try:
                embeds = embed_text(embed_list, "CLASSIFICATION", "text-multilingual-embedding-preview-0409")
                #embeddings.append(embeds)
            except:
                print("---- EMBEDDING FAILED ----")
                print(sentences[i])
                print(embed_list)
        embed_id = 0
        for word in sentences[i]:
            if word in dict_embeds:
                complete_embed_list.append(dict_embeds[word])
            else:
                complete_embed_list.append(embeds[embed_id])
                dict_embeds[word] = embeds[embed_id]
                embed_id += 1
        embeddings.append(complete_embed_list)
    embed_sentence = 0
    embed_word = 0
    for i in range(0, len(nodes['word'])):
        current_uri = nodes['word'][i]['uri'].split("_")[1:3]
        if i == 0:
            check_uri = current_uri[1]
        elif check_uri != current_uri[1]:
            embed_sentence += 1
            embed_word = 0
            check_uri = current_uri[1]
        if 'embeddings' not in nodes['word'][i]:
            nodes['word'][i]['embeddings'] = torch.Tensor(embeddings[embed_sentence][embed_word])
            embed_word += 1
    return nodes

def fetch_word_node_info(word, config_data):
    conection_string = config_data['connection'][0]['connection_uri']
    base_uri = config_data['connection'][0]['base_uri']
    graph_name = config_data['connection'][0]['graph_name']
    ntx = BuildNetworkx(base_uri, graph_name, conection_string)
    word_dict = ntx.fetch_word_info(word)
    return word_dict

def get_graph(list_to_get, config_data, test=False, targets_test = None, embedding = True):
    graph_data = HeteroData()
    edges = {}
    nodes = {}
    indexes = {}
    edges, nodes, indexes = fetch_graph(edges, nodes, indexes, list_to_get, config_data, test)
    if embedding:
        nodes = match_embeds_to_words(nodes)
    mapped_uris = map_uri_to_index(indexes, config_data)
    if test == 'Test':
        graph_data, unique_targets, size_targets = build_graph(nodes, edges, mapped_uris, config_data, graph_data, test=test, test_targets=targets_test, embedding=embedding)
    elif test == 'Train':
        graph_data, unique_targets, size_targets = build_graph(nodes, edges, mapped_uris, config_data, graph_data, test=test, embedding=embedding)
    else: # Benchmarks
        graph_data = build_graph(nodes, edges, mapped_uris, config_data, graph_data,
                                                               test=test, embedding=embedding)
    print(graph_data)
    print("--- DATASET LOADED AND TRANSFORMED ---")
    if test == 'Benchmark':
        return graph_data, mapped_uris, indexes['word']

    return graph_data, unique_targets, mapped_uris, indexes['word']