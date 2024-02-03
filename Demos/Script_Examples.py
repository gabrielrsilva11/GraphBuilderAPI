import pprint
import string
from rdflib import Graph
from rdflib.namespace import RDF, OWL
from Helpers import *
from InsertData import CreateGraph
from Query_Builder import *

base_uri = "http://ieeta-bit.pt/wikiner#"
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


uri_dict = {"http://ieeta-bit.pt/wikiner#": "ontoud"}
graph_name = "WikiNER"
qb = QueryBuilder(base_uri, graph_name)
conection_string = 'http://localhost:8890/sparql'

g = Graph()

relations_uri = {"http://ieeta-bit.pt/wikiner#" : "ieeta"}
connection = 'http://estga-fiware.ua.pt:8890/sparql'
# graph = CreateGraph(folder="DemoData", relations_uri=relations_uri,
#                   connection_string=connection, language='pt')
#
# # create_graph = CreateGraph(folder="DemoData")
pp = pprint.PrettyPrinter(indent=4)
# #g = build_subgraph(g, qb.build_query_by_sentence_id(500), conection_string)
# info_list = []
#
#
# senttext_uri = URIRef("http://ieeta.pt/ontoud#senttext")
# aa = qb.build_query_by_and_sentence_list(['Portugal', 'Algarve', 'Europa'])
# print(aa)
# #
# for sent_id in fetch_id_by_sentence(aa, conection_string):
#     g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), conection_string)
#
# for s, p, o in g.triples((None, senttext_uri, None)):
#     print(s, p, o)
#
# nodes_list = [edge_uri, id_uri, word_uri, lemma_uri]
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     print(s,p,o)
#     grafo = list_subgraph(nodes_list = nodes_list, graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri)
#     pp.pprint(grafo)

for sent_id in fetch_id_by_sentence(qb.build_query_by_sentence_start("A Biblia contem um numero de "), conection_string):
    print(sent_id)
    g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), conection_string)

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
#         print("Final text:", final_text)
#         for word in final_text:
#             wiki_id, titles = fetch_wiki_data(word)
#             if wiki_id:
#                 # graph.insert_wikimapper_data(i, titles)
#                 pass
#         print(' '.join(final_text))
#         wiki_id, titles = fetch_wiki_data(' '.join(final_text))
#         if wiki_id:
#             # graph.insert_wikimapper_data(i, titles)
#             pass
# words = "desde meados de"
# words = words.split(" ")
# #
# for i in range(0, 999):
#     g = Graph()
#     g = build_subgraph(g, qb.build_query_by_sentence_id(i), conection_string)
#     for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#         results_list = []
#         for word in words:
#             text = []
#             find_word_node(graph = g, root_node = s, transverse_by = depgraph_uri, order_by = id_uri,
#                               stop_word = word, result = text, word_uri= word_uri)
#             if text:
#                 dependencies = node_to_dependencies(graph = g, root_node = text[0],
#                                                      transverse_by = head_uri, order_by = id_uri)
#                 pp.pprint(dependencies)
#                 results_list.append(filter_dependencies(dependencies, 'word', filter_root=True))
#         if results_list:
#             for results in results_list:
#                 results.sort(key=lambda x: int(x[1]))
#                 results = [x[0] for x in results]
#                 for word in results:
#                     id = print(fetch_wiki_data(word))
#                     if id:
#                         pass
#                         #graph.insert_wikimapper_data(i, id)
#                 print(' '.join(results))
#                 print(fetch_wiki_data(' '.join(results)))
# print(set.intersection(*[set(x) for x in results]))

#
#
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     grafo = list_subgraph(nodes_list = nodes_list, graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri)
#     pp.pprint(grafo)

