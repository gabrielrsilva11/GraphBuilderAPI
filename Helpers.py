from rdflib import Graph, URIRef, Literal, BNode, Namespace, term
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper2
from wikimapper import WikiMapper
import re
from rdflib.namespace import RDFS, RDF


def list_subgraph(graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                  order_by: URIRef = None) -> list:
    """

    :param graph:
    :param root_node:
    :param transverse_by:
    :param order_by:
    :return:
    """
    thisnod, subnods, edge = [], [],''
    edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
    id_uri = URIRef("http://ieeta.pt/ontoud#id")
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    #print("Root: ", root_node)
    for s, p, o in graph.triples((root_node, None, None)):
        #print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            subnods.append(
                list_subgraph(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
        elif p == id_uri:
            thisnod.insert(0, o.toPython())
        elif p == edge_uri:
            edge = o.toPython()
        elif p == word_uri: #informação a recolher no nodo
            thisnod.append(o.toPython())
    if thisnod:
        subnods.append(thisnod)
    subnods.sort()
    # por questões de ordenação adicionamos neste nível o id_uri ao início da lista, por isso retiramo-lo
    # dos subnods recebidos do nível anterior
    if not thisnod:
        thisnod = [0]
    #return [thisnod[0], [x for _,x in subnods]]
    #return [thisnod[0], *subnods]
    return [thisnod[0], edge, *subnods]


def word_to_dependencies(graph: Graph = None, root_node: URIRef = None, trasverse_by: URIRef = None,
                         order_by: URIRef = None) -> list:
    """

    :param graph:
    :param root_node:
    :param trasverse_by:
    :param order_by:
    :return:
    """
    thisnod, subnods, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        # print(f"{s}\t{p}\t{o}")
        if p == trasverse_by:
            subnods.append(
                word_to_dependencies(graph=graph, root_node=o, trasverse_by=trasverse_by, order_by=order_by))
        else:
            labl = re.search(r".*#(\w+)", p.toPython())
            thisnod[labl.group(1)] = o.toPython()
    return [thisnod, *subnods]


def find_word_node(graph: Graph = None, root_node: URIRef = None, trasverse_by: URIRef = None,
                   order_by: URIRef = None, stop_word: str = None, result: list = []) -> list:
    """

    :param graph:
    :param root_node:
    :param trasverse_by:
    :param order_by:
    :param stop_word:
    :param result:
    :return:
    """
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    thisnod, subnods, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        if p == trasverse_by:
            subnods.append(
                find_word_node(graph=graph, root_node=o, trasverse_by=trasverse_by, order_by=order_by, stop_word=stop_word,
                               result=result))
        elif p == word_uri:
            word = o.toPython()
            if o.toPython() == stop_word:
                # if word == stop_word:
                result.append(s)
    return subnods


def check_for_edges(g: Graph = None, edges : list = []):
    """

    :param g:
    :param edges:
    :return:
    """
    edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
    root_nodes = []
    for s, p, o in g.triples((None, edge_uri, None)):
        for edge in edges:
            if o.toPython() == edge:
                root_nodes.append(s)
    return root_nodes


def find_edge_node(graph: Graph = None, root_node: URIRef = None, trasverse_by: URIRef = None,
                   order_by: URIRef = None, stop_node: str = None, result: list = []) -> list:
    """

    :param graph:
    :param root_node:
    :param trasverse_by:
    :param order_by:
    :param stop_node:
    :param result:
    :return:
    """
    edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
    thisnod, subnods, edge = defaultdict(), [], ''
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    id_uri = URIRef("http://ieeta.pt/ontoud#id")
    both = []
    for s, p, o in graph.triples((root_node, None, None)):
        if p == word_uri:
            result.append(o.toPython())
        elif p == id_uri:
            result.append(o.toPython())
        elif p == trasverse_by:
            subnods.append(
                find_edge_node(graph=graph, root_node=o, trasverse_by=trasverse_by, order_by=order_by, stop_node=stop_node,
                               result=result))
        # elif p == edge_uri:
        #     if o.toPython() == stop_node:
        #         # if word == stop_word:
        #         result.append(s)
    return subnods


def filter_dependencies(dependencies: list, attribute: str, filter_root: bool) -> list:
    """

    :param dependencies:
    :param attribute:
    :param filter_root:
    :return:
    """
    path = []
    if len(dependencies) == 2:
        path = filter_dependencies(dependencies[1], attribute, filter_root)
    if filter_root:
        if dependencies[0]['edge'] != 'root':
            path.append([dependencies[0][attribute], dependencies[0]['id']])
    else:
        path.append([dependencies[0][attribute], dependencies[0]['id']])
    return path


def build_subgraph(g: Graph, query: str, connection: str):
    """

    :param g:
    :param query:
    :param connection:
    :return:
    """
    sparql = SPARQLWrapper2(connection)
    #print(query)
    sparql.setQuery(query)
    for result in sparql.query().bindings:
        g.add(map(lambda x: URIRef(x.value) if x.type == 'uri' else
                (Literal(int(x.value)) if x.type == 'typed-literal' else Literal(x.value)), result.values()))
    return g


def fetch_id_by_sentence(query: str, connection: str):
    """

    :param query:
    :param connection:
    :return:
    """
    sparql = SPARQLWrapper2(connection)
    sparql.setQuery(query)
    for result in sparql.query().bindings:
        #print(result['s'].value, result['st'].value)
        yield result['s'].value


def fetch_wiki_data(text: str):
    """

    :param text:
    :return:
    """
    mapper = WikiMapper("data/index_ptwiki-latest.db")
    id = mapper.title_to_id(text)
    titles = mapper.id_to_titles(id)
    return id, titles

