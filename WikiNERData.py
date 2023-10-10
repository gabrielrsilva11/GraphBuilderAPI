from Query_Builder import *
from Helpers import *
from rdflib import Graph
from rdflib.namespace import RDF
from unidecode import unidecode
from spacy_conll import init_parser
import os
from tqdm import tqdm
import warnings
import pprint
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=UserWarning)
pp = pprint.PrettyPrinter(indent=4)

def add_ner_data(nlp):
    with tqdm(total=os.path.getsize(file_path)) as pbar:
        with open(file_path) as file:
            for line in file:
                if line != "\n":
                    sentence_start = ""
                    sentence = ""
                    entity_list = []
                    final_list = []
                    split_spaces = line.split(" ")
                    for i in range(0, len(split_spaces)):
                        split_bars = split_spaces[i].split("|")
                        sentence_start = sentence_start + split_bars[0] + " "
                        if split_bars[2].strip() != "O":
                            entity_list.append([split_bars[0], split_bars[2], i + 1])

                    doc = nlp(sentence_start)
                    conll = doc._.pandas
                    for index, row in conll.iterrows():
                        word = row['form'].replace("'", "").replace("\"", "")
                        word_id = row['id']
                        sentence += unidecode(word) + " "
                        if entity_list:
                            for i in range(0, len(entity_list)):
                                if word == entity_list[i][0] and word_id >= entity_list[i][2]:
                                    final_list.append([word, word_id, entity_list[i][1]])
                                    del entity_list[i]
                                    break

                    sentence = sentence.strip()
                    for sent_id in fetch_id_by_sentence(qb.build_query_by_sentence_start(sentence), connection_string):
                        if sent_id:
                            split_sent = sent_id.split("_")
                            for entities in final_list:
                                final_word_uri = word_uri + "_" + split_sent[-2] + "_" + split_sent[-1] + "_" + str(
                                    entities[1])
                                insert_data(final_word_uri, wikinerentity_uri, Literal(entities[2]), base_uri,
                                            graph_name, connection_string)
                pbar.update(len(line.encode('utf-8')))


def get_ner_ud_path(nlp):
    g = Graph()
    path_list = []
    path_dict = defaultdict()
    path_counter = defaultdict()
    path_counter_int = 1
    with open(file_path) as file:
        for line in file:
            if line != "\n":
                sentence = ""
                sentence_start = ""
                entity_list = []
                split_spaces = line.split(" ")
                for i in range(0, len(split_spaces)):
                    split_bars = split_spaces[i].split("|")
                    sentence_start = sentence_start + split_bars[0] + " "
                    if split_bars[2].strip() != "O":
                        entity_list.append(split_bars[0])
                print(sentence_start)
                doc = nlp(sentence_start)
                conll = doc._.pandas
                for index, row in conll.iterrows():
                    word = row['form'].replace("'", "").replace("\"", "")
                    word_id = row['id']
                    sentence += unidecode(word) + " "
                sentence = sentence.strip()
                for sent_id in fetch_id_by_sentence(qb.build_query_by_sentence_start(sentence), connection_string):
                    if sent_id:
                        g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), connection_string)
                        for s, p, o in g.triples((None, RDF.type, sentence_uri)):
                            for ner_word in entity_list:
                                text = []
                                find_word_node(graph = g, root_node = s, transverse_by = depgraph_uri, order_by = id_uri,
                                                  stop_word = ner_word, result = text, word_uri= word_uri)
                                if text:
                                    dependencies = node_to_dependencies(graph = g, root_node = text[0],
                                                                         transverse_by = head_uri, order_by = id_uri)
                                    #pp.pprint(dependencies)
                                    #path_list.append(filter_dependencies(dependencies, 'edge', filter_root=False, filter_id=True))
                                    dependency_list = filter_dependencies(dependencies, 'edge', filter_root=False, filter_id=True)
                                    if dependency_list in path_dict.values():
                                        value = {j for j in path_dict if path_dict[j] == dependency_list}
                                        for k in value:
                                            path_counter[k] += 1
                                    else:
                                        path_dict[path_counter_int] = dependency_list
                                        path_counter[path_counter_int] = 1
                                        path_counter_int += 1
    path_file = open("paths_file.txt", "w")
    for key in path_counter:
        file_writing = str(path_dict[key]) + ":" + str(path_counter[key]) + "\n"
        path_file.write(file_writing)
    path_file.close()


base_uri = "http://ieeta-bit.pt/wikiner#"
uri_dict = {"http://ieeta-bit.pt/wikiner#": "ontoud"}
graph_name = "WikiNER"
qb = QueryBuilder(base_uri, graph_name)
connection_string = 'http://localhost:8890/sparql'
file_path = "WikiNER_Original/aij-wikiner-pt-wp3"

sentence_uri = URIRef(base_uri + "Sentence")
head_uri = URIRef(base_uri + "head")
word_uri = URIRef(base_uri + "word")
senttext_uri = URIRef(base_uri + "senttext")
edge_uri = URIRef(base_uri + "edge")
pos_uri = URIRef(base_uri + "pos")
feats_uri = URIRef(base_uri + "feats")
id_uri = URIRef(base_uri + "id")
lemma_uri = URIRef(base_uri + "lemma")
poscoarse_uri = URIRef(base_uri + "poscoarse")
depgraph_uri = URIRef(base_uri + "depGraph")
wikinerentity_uri = URIRef(base_uri + "wikinerEntity")

nlp = init_parser("pt",
                  "stanza",
                  ext_names={"conll_pd": "pandas"},
                  parser_opts={"use_gpu": True, "verbose": False},
                  include_headers=True)

get_ner_ud_path(nlp)