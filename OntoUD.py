from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph
from wikimapper import WikiMapper

from Query_Builder import *


class OntoUD:
    def __init__(self, connection: str, base_uri: str):
        """
        Class of helper functions to build different subgraphs based on the user needs.
        :param connection: Connection to a triple-storage database to fetch the data
        """
        self.edge_uri = URIRef(base_uri + "#edge")
        self.id_uri = URIRef(base_uri + "#id")
        self.word_uri = URIRef(base_uri + "#word")
        self.sparql = SPARQLWrapper2(connection)

    def list_subgraph(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                      order_by: URIRef = None) -> list:
        """
        Expands a node and builds a subgraph of all its related nodes.
        :param graph: knowledge graph from which to build the subnode.
        :param root_node: node from which to start expanding and building the subgraph.
        :param transverse_by: URI that we want to expand from. Ex: head_uri
        :param order_by: ordering purposes for returning the subgraph.
        :return:
        """
        this_node, sub_nodes, edge = [], [], ''
        for s, p, o in graph.triples((root_node, None, None)):
            # print(f"{s}\t{p}\t{o}")
            if p == transverse_by:
                sub_nodes.append(
                    self.list_subgraph(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
            elif p == self.id_uri:
                this_node.insert(0, o.toPython())
            elif p == self.edge_uri:
                edge = o.toPython()
            elif p == self.word_uri:  # Information to keep on the node
                this_node.append(o.toPython())
        if this_node:
            sub_nodes.append(this_node)
        sub_nodes.sort()
        # Due to ordering we append the id_uri at the start of the list, therefore we take it out of the sub_nodes
        # previously appended
        if not this_node:
            this_node = [0]

        return [this_node[0], edge, *sub_nodes]

    def word_to_dependencies(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                             order_by: URIRef = None) -> list:
        """

        :param graph: knowledge graph from which to build the subnode.
        :param root_node: node from which to start expanding and building the subgraph.
        :param transverse_by: URI that we want to expand from. Ex: head_uri
        :param order_by: ordering purposes for returning the subgraph.
        :return:
        """
        this_node, sub_nodes, edge = defaultdict(), [], ''
        for s, p, o in graph.triples((root_node, None, None)):
            # print(f"{s}\t{p}\t{o}")
            if p == transverse_by:
                sub_nodes.append(
                    self.word_to_dependencies(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
            else:
                label = re.search(r".*#(\w+)", p.toPython())
                this_node[label.group(1)] = o.toPython()
        return [this_node, *sub_nodes]

    def find_word_node(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                       order_by: URIRef = None, stop_word: str = None, result: list = []) -> list:
        """

        :param graph: knowledge graph from which to build the subnode.
        :param root_node: node from which to start expanding and building the subgraph.
        :param transverse_by: URI that we want to expand from. Ex: head_uri
        :param order_by: ordering purposes for returning the subgraph.
        :param stop_word: word to find in the subgraph
        :param result:
        :return:
        """
        this_node, sub_nodes, edge = defaultdict(), [], ''
        for s, p, o in graph.triples((root_node, None, None)):
            if p == transverse_by:
                sub_nodes.append(
                    self.find_word_node(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by,
                                        stop_word=stop_word, result=result))
            elif p == self.word_uri:
                if o.toPython() == stop_word:
                    result.append(s)
        return sub_nodes

    def fetch_path(self, dependencies: list) -> list:
        """

        :param dependencies: a list of dependencies outputted from the list_subgraph function
        :return: the path in the existing graph.
        """
        path = []
        if len(dependencies) == 2:
            path = self.fetch_path(dependencies[1])
            path.append(dependencies[0]['edge'])
        else:
            path.append(dependencies[0]['edge'])
        return path

    def build_subgraph(self, g: Graph, query: str):
        """
        Builds a subgraph from a sparql query. This subgraph can then be used in the other functions.
        :param g: graph object used to append the subgraph.
        :param query: query from which to build the subgraph.
        :return: a graph object with a built subgraph
        """
        self.sparql.setQuery(query)
        for result in self.sparql.query().bindings:
            g.add(map(lambda x: URIRef(x.value) if x.type == 'uri' else (Literal(int(x.value))
                                                                         if x.type == 'typed-literal'
                                                                         else Literal(x.value)), result.values()))
        return g

    def fetch_id_by_sentence(self, query: str, connection: str):
        """

        :param query:
        :param connection:
        :return:
        """
        self.sparql.setQuery(query)
        for result in self.sparql.query().bindings:
            yield result['s'].value

    def fetch_wiki_data(self, text: str):
        """

        :param text:
        :return:
        """
        mapper = WikiMapper("data/index_ptwiki-latest.db")
        wiki_id = mapper.title_to_id(text)
        titles = mapper.id_to_titles(wiki_id)
        return wiki_id, titles