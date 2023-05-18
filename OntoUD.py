from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper2
from rdflib import Graph
from Query_Builder import *


class OntoUD:
    def __init__(self, connection: str):
        """

        :param connection:
        """
        self.edge_uri = URIRef("http://ieeta.pt/ontoud#edge")
        self.id_uri = URIRef("http://ieeta.pt/ontoud#id")
        self.word_uri = URIRef("http://ieeta.pt/ontoud#word")
        self.sparql = SPARQLWrapper2(connection)

    def list_subgraph(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
                      order_by: URIRef = None) -> list:
        """

        :param graph:
        :param root_node:
        :param transverse_by:
        :param order_by:
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
        # return [this_node[0], [x for _,x in sub_nodes]]
        # return [this_node[0], *sub_nodes]
        return [this_node[0], edge, *sub_nodes]

    def word_to_dependencies(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
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
                    self.word_to_dependencies(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
            else:
                label = re.search(r".*#(\w+)", p.toPython())
                this_node[label.group(1)] = o.toPython()
        return [this_node, *sub_nodes]

    def find_word_node(self, graph: Graph = None, root_node: URIRef = None, transverse_by: URIRef = None,
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

        :param dependencies:
        :return:
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

        :param g:
        :param query:
        :return:
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
