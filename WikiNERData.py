from Query_Builder import *
from Helpers import *
from rdflib import Graph
from unidecode import unidecode
from spacy_conll import init_parser
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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

with tqdm(total=os.path.getsize(file_path)) as pbar:
    with open(file_path) as file:
        for line in file:
            if line != "\n":
                g = Graph()
                sentence_start = ""
                sentence = ""
                entity_list = []
                final_list = []
                split_spaces = line.split(" ")
                for i in range(0, len(split_spaces)):
                    split_bars = split_spaces[i].split("|")
                    sentence_start = sentence_start + split_bars[0] + " "
                    if split_bars[2].strip() != "O":
                        entity_list.append([split_bars[0], split_bars[2], i+1])

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
                            final_word_uri = word_uri+"_"+split_sent[-2]+"_"+split_sent[-1]+"_"+str(entities[1])
                            insert_data(final_word_uri, wikinerentity_uri, Literal(entities[2]), base_uri, graph_name, connection_string)
            pbar.update(len(line.encode('utf-8')))
                # for sent_id in fetch_id_by_sentence(qb.build_query_by_sentence_start(sentence), connection_string):
                #     g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), connection_string)
                #
                # full_match = ""
                # for i in range(0, len(split_spaces)):
                #     split_bars = split_spaces[i].split("|")
                #     if split_bars[2].strip() != "O":
                #         full_match += split_bars[0] + " "
                #
                #         for s, p, o in g.triples((None, word_uri, Literal(split_bars[0]))):
                #             #insert_data(s, wikinerentity_uri, Literal(split_bars[2]), base_uri, graph_name, connection_string)
                #             previous_id = s.split("_")
                #             print(s)