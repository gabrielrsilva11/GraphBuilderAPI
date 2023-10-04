import re
from collections import defaultdict
from time import sleep
import rdflib.term
from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph, URIRef, Literal
from wikimapper import WikiMapper
from Query_Builder import QueryBuilder

def list_conll_subgraph(graph: Graph, root_node: rdflib.term.Node, transverse_by: URIRef, main_uri: str,
                        order_by: URIRef = None) -> list:
    """
    Creates a "subgraph" in the form of a list. This subgraph starts on a given root node and is transversed by a given
    URI until no more are found. The ID, Word and Edge information are returned along.

    :param graph: Graph that will be transversed
    :param root_node: Where to start searching the graph
    :param transverse_by: The URI used to search deeper within the graph
    :param order_by: What to order by the final list
    :return:
    List containing the nodes related to the root node as well as the edge, id and word of these nodes.
    """
    this_node, sub_nodes, edge = [], [], ''
    edge_uri = URIRef(main_uri + "edge")
    id_uri = URIRef(main_uri + "id")
    word_uri = URIRef(main_uri + "word")
    # print("Root: ", root_node)
    for s, p, o in graph.triples((root_node, None, None)):
        #print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            sub_nodes.append(
                list_conll_subgraph(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by, main_uri=main_uri))
        elif p == id_uri:
            this_node.insert(0, o.__str__())
        elif p == edge_uri:
            edge = o.__str__()
        elif p == word_uri:  # Information to collection on the node.
            this_node.append(o.__str__())
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


def list_subgraph(nodes_list: list, graph: Graph, root_node: rdflib.term.Node, transverse_by: URIRef,
                  order_by: URIRef = None) -> list:
    """
    Creates a "subgraph" in the form of a list. This subgraph starts on a given root node and is transversed by a given
    URI until no more are found. The information saved in this graph is related to the URIs seen in nodes_list.

    :param nodes_list: List of URIs whose information should be saved in the final list.
    :param graph: Graph that will be transversed
    :param root_node: Where to start searching the graph
    :param transverse_by: The URI used to search deeper within the graph
    :param order_by: What to order by the final list
    :return:
    List containing the nodes related to the root node as well as the information extracted from the URIs in the
    nodes_list.
    """
    this_node, sub_nodes, edge = [], [], ''
    # print("Root: ", root_node)
    for s, p, o in graph.triples((root_node, None, None)):
        # print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            sub_nodes.append(
                list_subgraph(nodes_list=nodes_list, graph=graph, root_node=o, transverse_by=transverse_by,
                              order_by=order_by))
        elif p in nodes_list:
            this_node.insert(0, o.__str__())
    if this_node:
        sub_nodes.append(this_node)
    sub_nodes.sort()
    # Due to ordering we append the id_uri at the start of the list, therefore we take it out of the sub_nodes
    # previously appended
    if not this_node:
        this_node = [0]
    # return [this_node[0], [x for _,x in sub_nodes]]
    # return [this_node[0], *sub_nodes]
    return [this_node[0],  *sub_nodes]


def node_to_dependencies(graph: Graph, root_node: rdflib.term.Node, transverse_by: URIRef,
                         order_by: URIRef = None) -> list:
    """
    Starts on a root node and will return a list of the given dependencies for that node. These dependencies are related
    to what the user wants to transverse the graph by.

    For example, in CoNLL, transversing by a Head URI will let the user know the path from the given node to the root
    of the sentence.

    :param graph: Graph that will be transversed
    :param root_node: Where to start searching the graph
    :param transverse_by: The URI used to search deeper within the graph
    :param order_by: What to order by the final list
    :return:
    List that starts on the given root node and finishes on the last node related to it by a given URI.
    """
    this_node, sub_nodes, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        # print(f"{s}\t{p}\t{o}")
        if p == transverse_by:
            sub_nodes.append(
                node_to_dependencies(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
        else:
            label = re.search(r".*#(\w+)", p.__str__())
            this_node[label.group(1)] = o.__str__()
    return [this_node, *sub_nodes]


def find_word_node(graph: Graph, root_node: rdflib.term.Node, transverse_by: URIRef, stop_word: str,
                   order_by: URIRef = None, result: list = None, word_uri: URIRef = None) -> list:
    """

    :param graph: Graph that will be transversed
    :param root_node: Where to start searching the graph
    :param transverse_by: The URI used to search deeper within the graph
    :param order_by: What to order by the final list
    :param stop_word: Word to find and stop transversing the graph
    :param word_uri:
    :param result: Appends the node where the stop word was found
    :return:
    The result of this function is put inside the "result" list which is passed as a parameter.
    """
    this_node, sub_nodes, edge = defaultdict(), [], ''
    for s, p, o in graph.triples((root_node, None, None)):
        if p == transverse_by:
            sub_nodes.append(
                find_word_node(graph=graph, root_node=o, transverse_by=transverse_by,
                               order_by=order_by, stop_word=stop_word, result=result))
        elif p == word_uri:
            if o.__str__() == stop_word:
                # if word == stop_word:
                result.append(s)
    return sub_nodes


def check_for_edges(g: Graph, edges: list = None, edge_uri: URIRef = None) -> list:
    """
    Extracts all the roots containing specific types of edges in a given graph.

    :param g: Graph in which to look for.
    :param edges: list of the edges to check.
    :param edge_uri: The URI of the edge variable.
    :return:
    Returns the nodes that contain the given edge relations.
    """
    root_nodes = []
    for s, p, o in g.triples((None, edge_uri, None)):
        for edge in edges:
            if o.__str__() == edge:
                root_nodes.append(s)
    return root_nodes


def find_edge_node(graph: Graph, root_node: rdflib.term.Node, transverse_by: URIRef, stop_node: str,
                   order_by: URIRef = None, result: list = None) -> list:
    """

    :param graph: Graph that will be transversed
    :param root_node: Where to start searching the graph
    :param transverse_by: The URI used to search deeper within the graph
    :param order_by: What to order by the final list
    :param stop_node: Word to find and stop transversing the graph
    :param result: Appends the node where the stop word was found. Appends the word and the id of the word into the list
    :return:
    The result of this function is put inside the "result" list which is passed as a parameter.
    :return:
    """
    this_node, sub_nodes, edge = defaultdict(), [], ''
    word_uri = URIRef("http://ieeta.pt/ontoud#word")
    id_uri = URIRef("http://ieeta.pt/ontoud#id")
    for s, p, o in graph.triples((root_node, None, None)):
        if p == word_uri:
            result.append(o.__str__())
        elif p == id_uri:
            result.append(o.__str__())
        elif p == transverse_by:
            sub_nodes.append(
                find_edge_node(graph=graph, root_node=o, transverse_by=transverse_by,
                               order_by=order_by, stop_node=stop_node, result=result))
        # elif p == edge_uri:
        #     if o.__str__() == stop_node:
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
        triple = list(map(lambda x: URIRef(x.value) if x.type == 'uri' else (Literal(int(x.value))
                                                                             if x.type == 'typed-literal'
                                                                             else Literal(x.value)), result.values()))
        g.add((triple[0], triple[1], triple[2]))
    return g


def fetch_id_by_sentence(query: str, connection: str):
    """

    :param query:
    :param connection:
    :return:
    """
    sparql = SPARQLWrapper2(connection)
    sparql.setQuery(query)
    try:
        for result in sparql.query().bindings:
            # print(result['s'].value, result['st'].value)
            yield result['s'].value
    except:
        print("------- Error on Query --------")
        print(query)
        print("Ignoring this query and proceeding.")
        yield False


def fetch_wiki_data(text: str):
    """

    :param text:
    :return:
    """
    mapper = WikiMapper("wikimapper_data/index_ptwiki-latest.db")
    wiki_id = mapper.title_to_id(text)
    titles = mapper.id_to_titles(wiki_id)
    return wiki_id, titles


def insert_data(s, p, o, uri, graph_name, connection_string):
    """
    Inserts a triple into a triple-storage and a knowledge graph.

    :param s: subject of the triple
    :param p: predicate of the triple
    :param o: object of the triple
    :param connection_string: connection to the triple storage
    :return:
    """
    queries = QueryBuilder(main_uri=uri, graph_name=graph_name)
    wrapper = SPARQLWrapper2(connection_string)
    query = queries.build_insert_query(s, p, o)
    wrapper.setQuery(query)
    wrapper.method = 'POST'
    for i in range(0, 10):
        try:
            results = wrapper.query()
            str_error = None
        except:
            str_error = "Error"
            pass

        if str_error:
            print("Error occurred. Attempting to upload triple again.")
            print("Attempt number: ", i)
            sleep(2*i)  # wait for 2*attempt number seconds before trying to fetch the data again
        else:
            break