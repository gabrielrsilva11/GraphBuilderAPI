from Helpers import *
from Query_Builder import *
import pprint

conection_string = 'http://localhost:8890/sparql'
senttext_uri = URIRef("http://demo-ieeta.pt/ontoud#senttext")
depgraph_uri = URIRef("http://demo-ieeta.pt/ontoud#depGraph")
id_uri = URIRef("http://demo-ieeta.pt/ontoud#id")
main_uri = "http://demo-ieeta.pt/ontoud#"
qb = QueryBuilder(main_uri)
g = Graph()

query = qb.build_query_by_and_sentence_list(['Tropico', 'America', 'Sul'])
print(query)

for sent_id in fetch_id_by_sentence(query, conection_string):
    g = build_subgraph(g, qb.build_query_by_sentence_id(sent_id), conection_string)

for s, p, o in g.triples((None, senttext_uri, None)):
    print(s, p, o)
    grafo = list_conll_subgraph(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri, main_uri=main_uri)
    pprint.pprint(grafo)
    break
