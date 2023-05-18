import re
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph, URIRef, Literal
from wikimapper import WikiMapper


def list_subgraph(graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                  order_by: URIRef = None) -> list:
    """

    :param graph:
    :param root_node:
    :param transverse_by:
    :param order_by:
    :return:
    """
    this_node, sub_nodes, edge = [], [], ''
    edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
    id_uri = URIRef("http://ieeta.pt/ontoud#id")
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    # print("Root: ", root_node)
    for s, p, o in graph.triples((root_node, None, None)):
        # print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            sub_nodes.append(
                list_subgraph(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
        elif p == id_uri:
            this_node.insert(0, o.toPython())
        elif p == edge_uri:
            edge = o.toPython()
        elif p == word_uri:  # Information to collection on the node.
            this_node.append(o.toPython())
    if this_node:
        sub_nodes.append(this_node)
    sub_nodes.sort()
    # Due to ordering we append the id_uri at the start of the list, therefore we take it out of the sub_nodes
    # previously appended
    if not this_node:
        this_node = [0]
    # return [this_node[0], [x for _,x in sub_nodes]]
    # return [this_node[0], *sub_nodes]
    return [this_node[0], edge, *sub_nodes]


def word_to_dependencies(graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                         order_by: URIRef = None) -> list:
    """

    :param graph:
    :param root_node:
    :param transverse_by:
    :param order_by:
    :return:
    """
    this_node, sub_nodes, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        # print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            sub_nodes.append(
                word_to_dependencies(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
        else:
            label = re.search(r".*#(\w+)", p.toPython())
            this_node[label.group(1)] = o.toPython()
    return [this_node, *sub_nodes]


def find_word_node(graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                   order_by: URIRef = None, stop_word: str = None, result: list = []) -> list:
    """

    :param graph:
    :param root_node:
    :param transverse_by:
    :param order_by:
    :param stop_word:
    :param result:
    :return:
    """
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    this_node, sub_nodes, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        if p == transverse_by:
            sub_nodes.append(
                find_word_node(graph=graph, root_node=o, transverse_by=transverse_by,
                               order_by=order_by, stop_word=stop_word, result=result))
        elif p == word_uri:
            if o.toPython() == stop_word:
                # if word == stop_word:
                result.append(s)
    return sub_nodes


def check_for_edges(g: Graph = None, edges: list = []):
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


def find_edge_node(graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                   order_by: URIRef = None, stop_node: str = None, result: list = []) -> list:
    """

    :param graph:
    :param root_node:
    :param transverse_by:
    :param order_by:
    :param stop_node:
    :param result:
    :return:
    """
    this_node, sub_nodes, edge = defaultdict(), [], ''
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    id_uri = URIRef("http://ieeta.pt/ontoud#id")
    for s, p, o in graph.triples((root_node, None, None)):
        if p == word_uri:
            result.append(o.toPython())
        elif p == id_uri:
            result.append(o.toPython())
        elif p == transverse_by:
            sub_nodes.append(
                find_edge_node(graph=graph, root_node=o, transverse_by=transverse_by,
                               order_by=order_by, stop_node=stop_node, result=result))
        # elif p == edge_uri:
        #     if o.toPython() == stop_node:
        #         # if word == stop_word:
        #         result.append(s)
    return sub_nodes


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
    # print(query)
    sparql.setQuery(query)
    for result in sparql.query().bindings:
        g.add(map(lambda x: URIRef(x.value) if x.type == 'uri' else (Literal(int(x.value))
                                                                     if x.type == 'typed-literal'
                                                                     else Literal(x.value)), result.values()))
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
        # print(result['s'].value, result['st'].value)
        yield result['s'].value


def fetch_wiki_data(text: str):
    """

    :param text:
    :return:
    """
    mapper = WikiMapper("data/index_ptwiki-latest.db")
    wiki_id = mapper.title_to_id(text)
    titles = mapper.id_to_titles(wiki_id)
    return wiki_id, titles
