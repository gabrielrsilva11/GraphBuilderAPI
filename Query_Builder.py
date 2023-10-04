import re
from multipledispatch import dispatch
from rdflib import URIRef, Literal

class QueryBuilder:
    def __init__(self, main_uri: str, graph_name=""):
        """

        :param self_uri_dict: uris used on the graph
        :param graph_name: name of the graph to query
        """
        self.uri_dict = {'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
                         'http://www.w3.org/2002/07/owl#': 'owl'
                         }
        self.uri_dict.update({main_uri : 'main'})
        self.main_uri = main_uri
        self.graph_name = graph_name

    def build_query_by_sentence_start(self, start_string: str) -> str:
        """
        Builds a SPARQL query which will look for nodes that start with a given string
        :param start_string: String to look for in the nodes.
        :return: A SPARQL query String.
        """
        query = """
            select ?s ?st
            where {
            ?s <""" + self.main_uri + """senttext> ?st.
            FILTER(strstarts(?st, '""" + start_string + """')) .
            }
            LIMIT 1000 
        """
        return query

    def build_query_by_and_sentence_list(self, string_list: list) -> str:
        """
        Builds a SPARQL query to look for sentences that are present in a given list. This query will look for a
        sentences that contain ALL the strings in the list (AND query).
        :param string_list: list of strings to look for.
        :return: A SPARQL query String.
        """

        query = """
                select ?s ?st
                where {
                ?s <""" + self.main_uri + """senttext> ?st. 
                FILTER(contains(?st, '""" + string_list[0] + "')"

        for word in string_list[1:]:
            query = query + " && contains(?st, '""" + word + """')"""

        query = query + """)
                }
                LIMIT 1000"""
        return query

    def build_query_by_or_sentence_list(self, string_list: list) -> str:
        """
        Builds a SPARQL query to look for sentences that are present in a given list. This query will look for a
        sentences that contain at least one of the strings in the list (OR query).
        :param string_list: list of strings to look for.
        :return: A SPARQL query String.
        """
        query = """
                select ?s ?st
                where {
                ?s <""" + self.main_uri + """senttext> ?st.  
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
        Builds a SPARQL query to look for sentences by a given id.
        :param str_id: (string) id of the sentence to look for.
        :return: A SPARQL query String.
        """
        query = """
        PREFIX dbp:  <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
        SELECT ?s ?p ?o
        WHERE {
            <""" + str_id + """> <"""+self.main_uri+"""depGraph>* ?s . 
            ?s ?p ?o .
        }
        """
        return query

    @dispatch(int)
    def build_query_by_sentence_id(self, str_id: int) -> str:
        """
        Builds a SPARQL query to look for sentences by a given id.
        :param str_id: (int) id of the sentence to look for.
        :return: A SPARQL query String.
        """
        query = """
        PREFIX dbp:  <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
        SELECT ?s ?p ?o
        WHERE {
            <http://ieeta.pt/ontoud#Sentence_""" + str(str_id) + """> <""" + self.main_uri + """depGraph>* ?s . 
            ?s ?p ?o .
        }
        """
        return query

    @dispatch(str, str, str)
    def build_insert_query(self, s, p, o) -> str:
        """
        Builds a SPARQL query to insert a triple into a graph.
        :param s: (string) subject of the triple
        :param p: (string) predicate of the triple
        :param o: (string) object of the triple
        :return: A SPARQL query String.
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
        Builds a SPARQL query to insert a triple into a graph.
        :param s: (string) subject of the triple
        :param p: (URIRef) predicate of the triple
        :param o: (string) object of the triple
        :return: A SPARQL query String.
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

    @dispatch(str, URIRef, Literal)
    def build_insert_query(self, s, p, o) -> str:
        """
        Builds a SPARQL query to insert a triple into a graph.
        :param s: (string) subject of the triple
        :param p: (string) predicate of the triple
        :param o: (Literal) object of the triple
        :return: A SPARQL query String.
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
        REVER ESTA AQUI QUE PODE SER SUBSTITUIDA POR UM INSERT_QUERY NORMAL.
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
