from Query_Builder import *
from Helpers import *
from rdflib import Graph
from rdflib.namespace import RDF
from collections import defaultdict
import pandas as pd
from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import torch
from transformers import BertTokenizer, BertModel
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
import pylab as plt
import timeit
from tqdm import tqdm

def get_path_count():
    sparql_query = """
    select ?s ?entity ?word where{
        graph <http://www.ieeta-bit.pt/Conll_Coref_train#> {
            ?s <http://www.ieeta-bit.pt/Conll_Coref_train#ner_tags> ?entity .
            ?s <http://www.ieeta-bit.pt/Conll_Coref_train#word> ?word .
        }
    }
    """
    df = pd.DataFrame(columns=['path', 'Total', 'Total Entity', 'B Entity'])
    total = 0
    base_uri = "http://www.ieeta-bit.pt/Conll_Coref_train#"
    graph_name = "http://www.ieeta-bit.pt/Conll_Coref_train#"
    sentence_uri = URIRef(base_uri + "Sentence")
    depgraph_uri = URIRef(base_uri + "depGraph")
    head_uri = URIRef(base_uri + "head")
    id_uri = URIRef(base_uri + "id")
    word_uri = URIRef(base_uri + "word")
    connection_string = "http://hlt.ieeta.pt:8890/sparql"
    path_dict = defaultdict()
    path_counter = defaultdict()
    path_counter_base = defaultdict()
    path_counter_total = defaultdict()
    path_counter_int = 1
    previous_id = ''
    g = Graph()
    for result in perform_query(sparql_query, connection_string):
        total += 1
        entity_value = result['entity'].value
        word_id_split = result['s'].value.split("_")
        sentence_id = sentence_uri + "_" + word_id_split[-3]+ "_" + word_id_split[-2]
        if sentence_id != previous_id:
            g = Graph()
            g = build_subgraph(g, qb.build_query_by_sentence_id(sentence_id.strip()), connection_string)
        for s, p, o in g.triples((None, RDF.type, sentence_uri)):
            text = []
            find_word_node(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri,
                           stop_word=result['word'].value, result=text, word_uri=word_uri)
            if text:
                dependencies = node_to_dependencies(graph=g, root_node=text[0],
                                                    transverse_by=head_uri, order_by=id_uri)

                dependency_list = filter_dependencies(dependencies, 'edge', filter_root=False,
                                                      filter_id=True)
                dependency_nouri = [item.split("#")[-1].lower() for item in dependency_list]

                if dependency_nouri in path_dict.values():
                    value = {j for j in path_dict if path_dict[j] == dependency_nouri}
                    for k in value:
                        path_counter[k] += 1
                        if entity_value == '1' or entity_value == '3' or entity_value == '5' or entity_value == '7':
                            path_counter_base[k] += 1
                        if entity_value != '0':
                            path_counter_total[k] += 1
                else:
                    path_dict[path_counter_int] = dependency_nouri
                    path_counter[path_counter_int] = 1
                    if entity_value == '1' or entity_value == '3' or entity_value == '5' or entity_value == '7':
                        path_counter_base[path_counter_int] = 1
                    else:
                        path_counter_base[path_counter_int] = 0

                    if entity_value != '0':
                        path_counter_total[path_counter_int] = 1
                    else:
                        path_counter_total[path_counter_int] = 0
                    path_counter_int += 1
        previous_id = sentence_id
    #path_file = open("../TTLs/paths_file_conll.txt", "w")
    path_file = "../TTLs/paths.xlsx"
    for key in path_counter:
        df.loc[len(df.index)] = [path_dict[key], path_counter[key], path_counter_total[key], path_counter_base[key]]
    df.to_excel(path_file)

def word_query(word_id):
    query = """
        select ?p ?o where{
        graph <http://www.ieeta-bit.pt/Conll_Coref_train#> {
            <"""+word_id+"""> ?p ?o .
        }
    }
    """
    return query


def get_attributes_path():
    sparql_query = """
    select ?s ?entity ?word where{
        graph <http://www.ieeta-bit.pt/Conll_Coref_train#> {
            ?s <http://www.ieeta-bit.pt/Conll_Coref_train#ner_tags> ?entity .
            ?s <http://www.ieeta-bit.pt/Conll_Coref_train#word> ?word .
            FILTER(?entity != '0') .
        } 
    }
    """
    connection_string = "http://hlt.ieeta.pt:8890/sparql"
    df = pd.DataFrame(columns=['Number', 'PosCoarse', 'Pos', 'Wikidata', 'Position'])
    for result in perform_query(sparql_query, connection_string):
        word_id = result['s'].value
        query = word_query(word_id)
        wikidataId = False
        for result in perform_query(query, connection_string):
            type = result['p'].value.split('#')[-1]
            if type == 'poscoarse':
                poscoarse_name = result['o'].value.split('#')[-1]
            elif type == 'pos':
                pos_name = result['o'].value.split('#')[-1]
            elif type == 'wikidataId':
                wikidataId = True
            elif type == 'number':
                number_name = result['o'].value.split('#')[-1]
            elif type == 'head':
                current_word = int(word_id.split("#")[-1].split("_")[-1])
                head_word = int(result['o'].value.split('#')[-1].split("_")[-1])
                # if current_word > head_word:
                #     position = -1
                # elif current_word < head_word:
                #     position = 1
                position = current_word - head_word
            elif type == 'fromSentence':
                position = 0
        df.loc[len(df.index)] = [number_name, poscoarse_name, pos_name, wikidataId, position]
    df_counts = df.groupby(df.columns.tolist(),as_index=False).size()
    df_counts.to_excel("../TTLs/attributes.xlsx")

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

def get_bert_embeddings(text, tokenizer, model):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        word_embeddings = outputs[2]
    print("Number of layers:", len(word_embeddings), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0
    print("Number of batches:", len(word_embeddings[layer_i]))
    batch_i = 0
    print("Number of tokens:", len(word_embeddings[layer_i][batch_i]))
    token_i = 0
    print("Number of hidden units:", len(word_embeddings[layer_i][batch_i][token_i]))
    return word_embeddings


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


def get_word_embeddings():
    query = """
    select ?sentence ?text
        where{
            graph <http://www.ieeta-bit.pt/Conll_Coref_train#> {
                ?sentence ?p <http://www.ieeta-bit.pt/Conll_Coref_train#Sentence> .
                ?sentence <http://www.ieeta-bit.pt/Conll_Coref_train#senttext> ?text .
            }
    }
    """
    connection_string = "http://hlt.ieeta.pt:8890/sparql"
    i = 0
    for result in perform_query(query, connection_string):
        sentence = result['text'].value.replace("  ", " ").split(" ")
        print(len(sentence))
        if len(sentence) > 1:
            embeds = embed_text(sentence, "CLASSIFICATION", "text-multilingual-embedding-preview-0409")
            i += 1
            break
    return embeds

def build_graph(ids):
    connection_string = "http://hlt.ieeta.pt:8890/sparql"
    df_connections = pd.DataFrame(columns=['source', 'target', 'attribute'])
    attributes_columns = ['id', 'lemma', 'word', 'wikidataId', 'chunk_tags', 'pos_tags', 'y']
    df_attributes = pd.DataFrame(columns=attributes_columns)
    for id in ids:
        print(id)
        embeds = []
        for result in perform_query(qb.build_query_senttext_by_id(0, id), connection_string):
            sentence = result['text'].value.replace("  ", " ").split(" ")
           # embeds = embed_text(sentence, "CLASSIFICATION", "text-multilingual-embedding-preview-0409")
        for result_depgraph in perform_query(qb.build_query_by_sentence_id(0, id), connection_string):
            type = result_depgraph['s'].value.split("#")[-1].split("_")[0]
            id = result_depgraph['s'].value.split("#")[-1]
            if type == 'word':
                attribute_type = result_depgraph['p'].value.split("#")[-1]
                attribute_value = result_depgraph['o'].value
                if attribute_type in attributes_columns:
                    #attributes_dict[id][attribute_type] = attribute_value
                    if id in df_attributes.values:
                        df_attributes.loc[df_attributes.index[df_attributes['id'] == id], attribute_type] = attribute_value
                    else:
                        df_attributes.loc[len(df_attributes.index), 'id'] = id
                elif attribute_type == 'ner_tags':
                    df_attributes.loc[df_attributes.index[df_attributes['id'] == id], 'y'] = attribute_value
                else:
                    if attribute_type != 'type':
                        df_connections.loc[len(df_connections.index)] = [id, result_depgraph['p'].value, attribute_value]

    df_attributes.fillna(0, inplace=True)
    # print(df_attributes.to_string())
    # print(df_connections.to_string())
    g = nx.from_pandas_edgelist(df_connections,'source', 'target', 'attribute')
    g.add_nodes_from((n, dict(d)) for n, d in df_attributes.iterrows())
    print(g)
    # pyg_graph = from_networkx(g)
    # print(pyg_graph)

start = timeit.default_timer()
base_uri = "http://www.ieeta-bit.pt/Conll_Coref_train#"
graph_name = "http://www.ieeta-bit.pt/Conll_Coref_train#"
qb = QueryBuilder(base_uri, graph_name)
build_graph([*range(6482, 7483, 1)])
stop = timeit.default_timer()

print('Time: ', (stop - start)/60)