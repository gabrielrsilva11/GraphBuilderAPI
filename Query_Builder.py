import re
from multipledispatch import dispatch
from rdflib import URIRef, Literal

class QueryBuilder:
    def __init__(self, self_uri_dict, graph_name):
        """

        :param self_uri_dict:
        :param graph_name:
        """
        self.uri_dict = {'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
                         'http://www.w3.org/2002/07/owl#': 'owl'}
        self.uri_dict.update(self_uri_dict)
        self.graph_name = graph_name

    def build_query_by_sentence_start(self, start_string: str) -> str:
        """

        :param start_string:
        :return:
        """
        query = """
            select ?s ?st
            where {
            ?s <http://ieeta.pt/ontoud#senttext> ?st.
            FILTER(strstarts(?st, '""" + start_string + """')) .
            }
            LIMIT 1000 
        """
        return query

    def build_query_by_and_sentence_list(self, string_list: list) -> str:
        """

        :param string_list:
        :return:
        """
        query = """
                select ?s ?st
                where {
                ?s <http://ieeta.pt/ontoud#senttext> ?st. 
                FILTER(contains(?st, '""" + string_list[0] + "')"

        for word in string_list[1:]:
            query = query + " && contains(?st, '""" + word + """')"""

        query = query + """)
                }
                LIMIT 1000"""
        return query

    def build_query_by_or_sentence_list(self, string_list: list) -> str:
        """

        :param string_list:
        :return:
        """
        query = """
                select ?s ?st
                where {
                ?s <http://ieeta.pt/ontoud#senttext> ?st. 
                FILTER(contains(?st, '""" + string_list[0] + "')"

        for word in string_list[1:]:
            query = query + " || contains(?st, '""" + word + """')"""

        query = query + """)
                }
                LIMIT 1000"""
        return query

    @dispatch(str)
    def build_query_by_sentence_id(self, str_id: str) -> str:
        """

        :param str_id:
        :return:
        """
        query = """
        PREFIX dbp:  <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ietud: <http://ieeta.pt/ontoud#>
    
        SELECT ?s ?p ?o
        WHERE {
            <""" + str_id + """> ietud:depgraph* ?s . 
            ?s ?p ?o .
        }
        """
        return query

    @dispatch(int)
    def build_query_by_sentence_id(self, str_id: int) -> str:
        """

        :param str_id:
        :return:
        """
        query = """
        PREFIX dbp:  <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ietud: <http://ieeta.pt/ontoud#>
    
        SELECT ?s ?p ?o
        WHERE {
            <http://ieeta.pt/ontoud#Sentence_""" + str(str_id) + """> ietud:depGraph* ?s . 
            ?s ?p ?o .
        }
        """
        return query

    @dispatch(str, str, str)
    def build_insert_query(self, s, p, o) -> str:
        """

        :param s:
        :param p:
        :param o:
        :return:
        """
        triple = []
        prefix_set = set()
        for item in [s, p, o]:
            uri_comps = re.search("(.*[#/])([^/]+)", item)
            prefix_var = self.uri_dict[uri_comps.group(1)]
            prefix_set.add("PREFIX " + prefix_var + ":<" + uri_comps.group(1) + ">")
            triple.append(prefix_var + ":" + uri_comps.group(2))

        query = "\n".join(prefix_set) + "\nINSERT DATA {GRAPH <" + self.graph_name + "> {" + " ".join(triple) + "}}"
        # print(query)
        return query

    @dispatch(str, URIRef, str)
    def build_insert_query(self, s, p, o) -> str:
        """

        :param s:
        :param p:
        :param o:
        :return:
        """
        triple = []
        prefix_set = set()
        for item in [s, p, o]:
            uri_comps = re.search("(.*[#/])([^/]+)", item)
            prefix_var = self.uri_dict[uri_comps.group(1)]
            prefix_set.add("PREFIX " + prefix_var + ":<" + uri_comps.group(1) + ">")
            triple.append(prefix_var + ":" + uri_comps.group(2))

        query = "\n".join(prefix_set) + "\nINSERT DATA {GRAPH <" + self.graph_name + "> {" + " ".join(triple) + "}}"
        # print(query)
        return query

    @dispatch(str, str, Literal)
    def build_insert_query(self, s, p, o) -> str:
        """

        :param s:
        :param p:
        :param o:
        :return:
        """
        triple = []
        prefix_set = set()
        for item in [s, p]:
            uri_comps = re.search("(.*[#/])([^/]+)", item)
            prefix_var = self.uri_dict[uri_comps.group(1)]
            prefix_set.add("PREFIX " + prefix_var + ":<" + uri_comps.group(1) + ">")
            triple.append(prefix_var + ":" + uri_comps.group(2))
        triple.append("'" + repr(o.toPython()).replace('\'', '') + "'")
        query = "\n".join(prefix_set) + "\nINSERT DATA {GRAPH <" + self.graph_name + "> {" + " ".join(triple) + "}}"
        # print(query)
        return query

    def build_insert_wikimapper_query(self, s, p, o) -> str:
        """

        :param s:
        :param p:
        :param o:
        :return:
        """
        triple = []
        prefix_set = set()
        for item in [s, p]:
            uri_comps = re.search("(.*[#/])([^/]+)", item)
            prefix_var = self.uri_dict[uri_comps.group(1)]
            prefix_set.add("PREFIX " + prefix_var + ":<" + uri_comps.group(1) + ">")
            triple.append(prefix_var + ":" + uri_comps.group(2))
        triple.append("'" + repr(o.toPython()).replace('\'', '') + "'")
        query = "\n".join(prefix_set) + "\nINSERT DATA {GRAPH <" + self.graph_name + "> {" + " ".join(triple) + "}}"
        return query
