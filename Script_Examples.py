import pprint
import string
from rdflib import Graph
from rdflib.namespace import RDF, OWL
from Helpers import *
from InsertData import CreateGraph
from Query_Builder import *

sentence_uri = URIRef("http://ieeta.pt/ontoud#Sentence")
head_uri = URIRef("http://ieeta.pt/ontoud#head")
word_uri = URIRef("http://ieeta.pt/ontoud#word")
senttext_uri = URIRef("http://ieeta.pt/ontoud#senttext")
edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
pos_uri = URIRef("http://ieeta.pt/ontoud#pos")
feats_uri = URIRef("http://ieeta.pt/ontoud#feats")
id_uri = URIRef("http://ieeta.pt/ontoud#id")
lemma_uri = URIRef("http://ieeta.pt/ontoud#lemma")
poscoarse_uri = URIRef("http://ieeta.pt/ontoud#poscoarse")
depgraph_uri = URIRef("http://ieeta.pt/ontoud#depGraph")

uri_dict = {"http://ieeta.pt/ontoud#": "ontoud"}
graph_name = "ieetapt_attempt6"
qb = QueryBuilder(uri_dict, graph_name)
conection_string = 'http://localhost:8890/sparql'

g = Graph()
create_graph = CreateGraph(doc="data/wikiner_subset.txt")
pp = pprint.PrettyPrinter(indent=4)
# g = build_subgraph(g, qb.build_query_by_sentence_id(500), conection_string)
info_list = []

# nodes_list = [edge_uri, id_uri, word_uri, lemma_uri]
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     grafo = list_subgraph(nodes_list = nodes_list, graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri)
#     pp.pprint(grafo)

# for sent_id in fetch_id_by_sentence(qb.build_query_by_sentence_start("A Espanha"), conection_string):
#     g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), conection_string)

# edges = ["obj", "nsubj"]
# for i in range(0, 9999):
#     g = Graph()
#     g = build_subgraph(g, qb.build_query_by_sentence_id(i), conection_string)
#     root_nodes = check_for_edges(g, edges, edge_uri = edge_uri)
#     for s in root_nodes:
#         text = []
#         find_edge_node(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri, result=text)
#         joint_info = [text[x:x + 2] for x in range(0, len(text), 2)]
#         joint_info.sort(key=lambda x: int(x[0]))
#         final_text_with_punct = [x[1] for x in joint_info]
#         final_text = [''.join(char for char in item if char not in string.punctuation)
#                       for item in final_text_with_punct]
#         # print("Final text:", final_text)
#         for word in final_text:
#             wiki_id, titles = fetch_wiki_data(word)
#             if wiki_id:
#                 create_graph.insert_wikimapper_data(i, titles)
#         # print(' '.join(final_text))
#         wiki_id, titles = fetch_wiki_data(' '.join(final_text))
#         if wiki_id:
#             create_graph.insert_wikimapper_data(i, titles)

words = "desde meados de"
words = words.split(" ")

for i in range(0, 100):
    g = Graph()
    g = build_subgraph(g, qb.build_query_by_sentence_id(i), conection_string)
    for s, p, o in g.triples((None, RDF.type, sentence_uri)):
        results_list = []
        for word in words:
            text = []
            print(word)
            find_word_node(graph = g, root_node = s, transverse_by = depgraph_uri, order_by = id_uri,
                              stop_word = word, result = text, word_uri= word_uri)
            print(text)
            #if text:
            #    dependencies = node_to_dependencies(graph = g, root_node = text[0],
            #                                          transverse_by = head_uri, order_by = id_uri)
            #    pp.pprint(dependencies)
#                 results_list.append(filter_dependencies(dependencies, 'word', filter_root=True))
#         if results_list:
#             for results in results_list:
#                 results.sort(key=lambda x: int(x[1]))
#                 results = [x[0] for x in results]
#                 print(results)
#                 for word in results:
#                     id = print(fetch_wiki_data(word))
#                     if id:
#                         create_graph.insert_wikimapper_data(i, id)
#                 print(' '.join(results))
#                 print(fetch_wiki_data(' '.join(results)))
# print(set.intersection(*[set(x) for x in results]))

# aa = qb.build_query_by_and_sentence_list(['Portugal', 'Algarve', 'Europa'])
#
# for sent_id in fetch_id_by_sentence(aa, conection_string):
#     g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), conection_string)
#
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     grafo = list_subgraph(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri)
#     pp.pprint(grafo)
