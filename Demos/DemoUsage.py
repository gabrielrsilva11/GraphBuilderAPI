from Helpers import *
from Query_Builder import *
import pprint
from rdflib.namespace import RDF
import string

conection_string = 'http://34.175.171.126:8890/sparql'
base_uri = "http://www.ieeta-bit.pt/OpenIE_s2_en#"
graph_name = "http://www.ieeta-bit.pt/OpenIE_s2_en#"
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

qb = QueryBuilder(base_uri, graph_name)
g = Graph()

#Build a graph with the first sentence
g = build_subgraph(g, qb.build_query_by_sentence_id(0, 1), conection_string)

# #Which attributes to fetch in our subgraph
# nodes_list = [edge_uri, id_uri, word_uri, lemma_uri]
# #Go through the graph and print each dependency graph starting from the Root Node
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     grafo = list_subgraph(nodes_list = nodes_list, graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri)
#     pprint.pprint(grafo)

# #List of edges to look for
# edges = ["obj", "nsubj"]
# root_nodes = check_for_edges(g, edges, edge_uri = edge_uri)
# print(root_nodes)
# # Starting on the nodes that have the above edges
# # Fetch the words from their dependency graph
# for s in root_nodes:
#     text = []
#     find_edge_node(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri, result=text, base_uri=base_uri)
#     joint_info = [text[x:x + 2] for x in range(0, len(text), 2)]
#     joint_info.sort(key=lambda x: int(x[0]))
#     final_text_with_punct = [x[1] for x in joint_info]
#     final_text = [''.join(char for char in item if char not in string.punctuation)
#                   for item in final_text_with_punct]
#     print("Final text:", final_text)

#Starting with a list of words and fetch the path from them to the root word
# words = ["however", "four", "lancets"]
# for s, p, o in g.triples((None, RDF.type, sentence_uri)):
#     results_list = []
#     for word in words:
#         text = []
#         find_word_node(graph = g, root_node = s, transverse_by = depgraph_uri, order_by = id_uri,
#                           stop_word = word, result = text, word_uri= word_uri)
#         if text:
#             dependencies = node_to_dependencies(graph = g, root_node = text[0],
#                                                  transverse_by = head_uri, order_by = id_uri)
#             pprint.pprint(dependencies)
#             results_list.append(filter_dependencies(dependencies, 'word', filter_root=True, filter_id=False))
#     #Sort by ID order
#     if results_list:
#         for results in results_list:
#             results.sort(key=lambda x: int(x[1]))
#             results = [x[0] for x in results]
#             print(' '.join(results))

for s, p, o in g.triples((None, senttext_uri, None)):
    print(s, p, o)
    grafo = list_conll_subgraph(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri, main_uri=base_uri)
    pprint.pprint(grafo)