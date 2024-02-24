from Helpers import *
from Query_Builder import *
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from multipledispatch import dispatch

class BuildNetworkx:
    def __init__(self, base_uri, graph_name, connection_string):
        self.base_uri = base_uri #"http://ieeta-bit.pt/ontoud#"
        self.graph_name = graph_name #"WikiNER"
        self.conection_string = connection_string #'http://estga-fiware.ua.pt:8890/sparql'
        self.qb = QueryBuilder(self.base_uri, self.graph_name)

    @dispatch(int, int)
    def fetch_graph(self, start, stop):
        g = Graph()
        doc_id = 1

        for i in range(start, stop):
            g = build_subgraph(g, self.qb.build_query_by_sentence_id(doc_id, i), self.conection_string)
        return g

    @dispatch(list)
    def fetch_graph(self, ids_list):
        g = Graph()
        doc_id = 0

        for i in ids_list:
            g = build_subgraph(g, self.qb.build_query_by_sentence_id(doc_id, i), self.conection_string)
        return g

    def fetch_graph_with_query(self, g, uri_type):
        g = build_subgraph(g, self.qb.query_by_type(uri_type), self.conection_string)
        return g

    def build_graph(self, g, start, stop):
        doc_id = 1

        #Fetch first 3 sentences
        for i in range(start, stop):
            g = build_subgraph(g, self.qb.build_query_by_sentence_id(doc_id, i), self.conection_string)
        return g

    def fetch_networkx(self):
        g = Graph()
        doc_id = 1

        #Fetch first 3 sentences
        for i in range(1, 10):
            g = build_subgraph(g, self.qb.build_query_by_sentence_id(doc_id, i), self.conection_string)

        networkx_graph = rdflib_to_networkx_multidigraph(g)
        return networkx_graph

    def fetch_entity_list(self):
        query = self.qb.query_count_entities()
        results = perform_query(query, self.conection_string)
        return results


    def fetch_worths_with_entities(self):
        query = self.qb.query_words_with_entities()
        results = perform_query(query, self.conection_string)
        return results


#URI Section
# base_uri = "http://ieeta-bit.pt/ontoud#"
# sentence_uri = URIRef(base_uri + "Sentence")
# head_uri = URIRef(base_uri + "head")
# word_uri = URIRef(base_uri + "word")
# senttext_uri = URIRef(base_uri + "senttext")
# edge_uri = URIRef(base_uri + "edge")
# pos_uri = URIRef(base_uri + "pos")
# feats_uri = URIRef(base_uri + "feats")
# id_uri = URIRef(base_uri + "id")
# lemma_uri = URIRef(base_uri + "lemma")
# poscoarse_uri = URIRef(base_uri + "poscoarse")
# depgraph_uri = URIRef(base_uri + "depGraph")
# uri_dict = {"http://ieeta-bit.pt/ontoud#": "ontoud"}
#
# #Connection Settings
# graph_name = "WikiNER"
# conection_string = 'http://estga-fiware.ua.pt:8890/sparql'
#
#
# qb = QueryBuilder(base_uri, graph_name)
# g = Graph()
# doc_id = 1
#
# #Fetch first 100 sentences
# for i in range(1, 3):
#     g = build_subgraph(g, qb.build_query_by_sentence_id(doc_id, i), conection_string)
#
# networkx_graph = rdflib_to_networkx_multidigraph(g)
# print(networkx_graph)
# #Plot the graph
# pos = nx.spring_layout(networkx_graph, scale=2)
# edge_labels = nx.get_edge_attributes(networkx_graph, 'r')
# nx.draw_networkx_edge_labels(networkx_graph, pos)
# nx.draw(networkx_graph, with_labels=False)
# plt.show()

# split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
# graph = split(networkx_graph)
#
# print(graph)