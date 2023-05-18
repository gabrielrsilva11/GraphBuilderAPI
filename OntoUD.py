from Query_Builder import *
from rdflib import Graph, URIRef, Literal
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper2


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
        thisnod, subnods, edge = [], [], ''
        for s, p, o in graph.triples((root_node, None, None)):
            # print(f"{s}\t{p}\t{o}")
            if p == transverse_by:
                subnods.append(
                    self.list_subgraph(graph=graph, root_node=o, transverse_by=transverse_by, order_by=order_by))
            elif p == self.id_uri:
                thisnod.insert(0, o.toPython())
            elif p == self.edge_uri:
                edge = o.toPython()
            elif p == self.word_uri:  # informação a recolher no nodo
                thisnod.append(o.toPython())
        if thisnod:
            subnods.append(thisnod)
        subnods.sort()
        # por questões de ordenação adicionamos neste nível o id_uri ao início da lista, por isso retiramo-lo
        # dos subnods recebidos do nível anterior
        if not thisnod:
            thisnod = [0]
        # return [thisnod[0], [x for _,x in subnods]]
        # return [thisnod[0], *subnods]
        return [thisnod[0], edge, *subnods]

    def word_to_dependencies(self, graph: Graph = None, root_node: URIRef = None, trasverse_by: URIRef = None,
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
                    self.word_to_dependencies(graph=graph, root_node=o, trasverse_by=trasverse_by, order_by=order_by))
            else:
                labl = re.search(r".*#(\w+)", p.toPython())
                thisnod[labl.group(1)] = o.toPython()
        return [thisnod, *subnods]

    def find_word_node(self, graph: Graph = None, root_node: URIRef = None, trasverse_by: URIRef = None,
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
        thisnod, subnods, edge = defaultdict(), [], ''
        for s, p, o in graph.triples((root_node, None, None)):
            if p == trasverse_by:
                subnods.append(
                    self.find_word_node(graph=graph, root_node=o, trasverse_by=trasverse_by, order_by=order_by,
                                   stop_word=stop_word,
                                   result=result))
            elif p == self.word_uri:
                if o.toPython() == stop_word:
                    result.append(s)
        return subnods

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
            g.add(map(lambda x: URIRef(x.value) if x.type == 'uri' else
            (Literal(int(x.value)) if x.type == 'typed-literal' else Literal(x.value)), result.values()))
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