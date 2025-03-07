from pprint import pprint
from itertools import chain, groupby
from time import sleep

graph2 = [0,
         '',
         [23,
          'http://www.ieeta-bit.pt/OpenIE_s2_en#ROOT',
          [12,
           'http://www.ieeta-bit.pt/OpenIE_s2_en#ccomp',
           [2,
            'http://www.ieeta-bit.pt/OpenIE_s2_en#nsubjpass',
            [1, 'http://www.ieeta-bit.pt/OpenIE_s2_en#det', [1, 'The']],
            [2, 'majority'],
            [3,
             'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
             [3, 'of'],
             [5,
              'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj',
              [4, 'http://www.ieeta-bit.pt/OpenIE_s2_en#det', [4, 'the']],
              [5, 'windows'],
              [7,
               'http://www.ieeta-bit.pt/OpenIE_s2_en#relcl',
               [6, 'http://www.ieeta-bit.pt/OpenIE_s2_en#advmod', [6, 'now']],
               [7, 'visible'],
               [8,
                'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
                [8, 'at'],
                [10,
                 'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj',
                 [9, 'http://www.ieeta-bit.pt/OpenIE_s2_en#compound', [9, 'Chartres']],
                 [10, 'Cathedral']]]]]]],
           [11, 'http://www.ieeta-bit.pt/OpenIE_s2_en#auxpass', [11, 'were']],
           [12, 'made'],
           [13, 'http://www.ieeta-bit.pt/OpenIE_s2_en#cc', [13, 'and']],
           [14,
            'http://www.ieeta-bit.pt/OpenIE_s2_en#conj',
            [14, 'installed'],
            [15,
             'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
             [15, 'between'],
             [16,
              'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj',
              [16, '1205'],
              [17, 'http://www.ieeta-bit.pt/OpenIE_s2_en#cc', [17, 'and']],
              [18, 'http://www.ieeta-bit.pt/OpenIE_s2_en#conj', [18, '1240']]]]]],
          [19, 'http://www.ieeta-bit.pt/OpenIE_s2_en#punct', [19, ',']],
          [20, 'http://www.ieeta-bit.pt/OpenIE_s2_en#advmod', [20, 'however']],
          [22,
           'http://www.ieeta-bit.pt/OpenIE_s2_en#nsubj',
           [21, 'http://www.ieeta-bit.pt/OpenIE_s2_en#nummod', [21, 'four']],
           [22, 'lancets']],
          [23, 'preserve'],
          [24,
           'http://www.ieeta-bit.pt/OpenIE_s2_en#dobj',
           [24, 'panels'],
           [25,
            'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
            [25, 'of'],
            [27,
             'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj',
             [26, 'http://www.ieeta-bit.pt/OpenIE_s2_en#amod', [26, 'Romanesque']],
             [27, 'glass'],
             [28,
              'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
              [28, 'from'],
              [31,
               'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj',
               [29, 'http://www.ieeta-bit.pt/OpenIE_s2_en#det', [29, 'the']],
               [30, 'http://www.ieeta-bit.pt/OpenIE_s2_en#amod', [30, '12th']],
               [31, 'century']]]]],
           [33,
            'http://www.ieeta-bit.pt/OpenIE_s2_en#relcl',
            [32, 'http://www.ieeta-bit.pt/OpenIE_s2_en#nsubj', [32, 'which']],
            [33, 'survived'],
            [35,
             'http://www.ieeta-bit.pt/OpenIE_s2_en#dobj',
             [34, 'http://www.ieeta-bit.pt/OpenIE_s2_en#det', [34, 'the']],
             [35, 'fire'],
             [36,
              'http://www.ieeta-bit.pt/OpenIE_s2_en#prep',
              [36, 'of'],
              [37, 'http://www.ieeta-bit.pt/OpenIE_s2_en#pobj', [37, '1195']]]]]],
          [38, 'http://www.ieeta-bit.pt/OpenIE_s2_en#punct', [38, '.']]]]

# word_ids = [20, 21, 22, 24, 25, 26, 27, 28, 29, 30]

"""
Sentence: Nolan Bushnell esta no conselho consultivo da Anti - AgingGames .
Options:
   Subject: Nolan Bushnell // AgingGames // 
   Predicate: Nolan Bushnell está // está no conselho consultivo da Anti - AgingGames // está . //  
   Object: no // conselho consultivo // no consultivo // consultivo da Anti - AgingGames // 
"""

"""
graph = [0,
 '',
 [2,
  'http://www.ieeta-bit.pt/CaRB#ROOT',
  [1, 'http://www.ieeta-bit.pt/CaRB#nsubj', [1, 'They']],
  [2, 'claim'],
  
----  
  [5,
   'http://www.ieeta-bit.pt/CaRB#xcomp',
   [3, 'http://www.ieeta-bit.pt/CaRB#aux', [3, 'to']],
   [4, 'http://www.ieeta-bit.pt/CaRB#aux', [4, 'have']],
   [5, 'busted'],
   [6,
    'http://www.ieeta-bit.pt/CaRB#dobj',
    [6, 'spirits'],
    [7, 'http://www.ieeta-bit.pt/CaRB#punct', [7, ',']],
    [8,
     'http://www.ieeta-bit.pt/CaRB#conj',
     [8, 'poltergeists'],
     [9, 'http://www.ieeta-bit.pt/CaRB#cc', [9, 'and']],
     [11,
      'http://www.ieeta-bit.pt/CaRB#conj',
      [10, 'http://www.ieeta-bit.pt/CaRB#amod', [10, 'other']],
      [11, 'spooks']]]],  
   [12,
    'http://www.ieeta-bit.pt/CaRB#prep',
    [12, 'in'],
    [13,
     'http://www.ieeta-bit.pt/CaRB#pobj',
     [13, 'hundreds'],
     [14,
      'http://www.ieeta-bit.pt/CaRB#prep',
      [14, 'of'],
      [15,
       'http://www.ieeta-bit.pt/CaRB#pobj',
       [15, 'houses'],
       [16,
        'http://www.ieeta-bit.pt/CaRB#prep',
        [16, 'around'],
        [18,
         'http://www.ieeta-bit.pt/CaRB#pobj',
         [17, 'http://www.ieeta-bit.pt/CaRB#det', [17, 'the']],
         [18, 'country']]]]]]]],
----
  [19, 'http://www.ieeta-bit.pt/CaRB#punct', [19, '.']]]]
"""

graph3 = [0,
     '',
     [5,
      '',
      [1, '', [1, ' ']],
      [2, '', [2, 'Category'], [3, '', [3, 'of'], [4, '', [4, 'sets']]]],
      [5, 'is'],
      [7,
       '',
       [6, '', [6, 'the']],
       [7, 'category'],
       [10,
        '',
        [9, '', [8, '', [8, 'whose']], [9, 'objects']],
        [10, 'are'],
        [11, '', [11, 'sets']]]],
      [12, '', [12, '.']]]]

# graph = [0, '',
#          [2, 'http://www.ieeta-bit.pt/CaRB#ROOT', [1, 'http://www.ieeta-bit.pt/CaRB#nsubj', [1, 'They']], [2, 'claim'],
#           [5, 'http://www.ieeta-bit.pt/CaRB#xcomp', [3, 'http://www.ieeta-bit.pt/CaRB#aux', [3, 'to']],
#            [4, 'http://www.ieeta-bit.pt/CaRB#aux', [4, 'have']], [5, 'busted'],
#            [6, 'http://www.ieeta-bit.pt/CaRB#dobj', [6, 'spirits'], [7, 'http://www.ieeta-bit.pt/CaRB#punct', [7, ',']],
#             [8, 'http://www.ieeta-bit.pt/CaRB#conj', [8, 'poltergeists'],
#              [9, 'http://www.ieeta-bit.pt/CaRB#cc', [9, 'and']],
#              [11, 'http://www.ieeta-bit.pt/CaRB#conj', [10, 'http://www.ieeta-bit.pt/CaRB#amod', [10, 'other']],
#               [11, 'spooks']]]], [12, 'http://www.ieeta-bit.pt/CaRB#prep', [12, 'in'],
#                                   [13, 'http://www.ieeta-bit.pt/CaRB#pobj', [13, 'hundreds'],
#                                    [14, 'http://www.ieeta-bit.pt/CaRB#prep', [14, 'of'],
#                                     [15, 'http://www.ieeta-bit.pt/CaRB#pobj', [15, 'houses'],
#                                      [16, 'http://www.ieeta-bit.pt/CaRB#prep', [16, 'around'],
#                                       [18, 'http://www.ieeta-bit.pt/CaRB#pobj',
#                                        [17, 'http://www.ieeta-bit.pt/CaRB#det', [17, 'the']], [18, 'country']]]]]]]],
#           [19, 'http://www.ieeta-bit.pt/CaRB#punct', [19, '.']]]]

graph = [0, '', [2, 'http://www.ieeta-bit.pt/CaRB#ROOT', [1, 'http://www.ieeta-bit.pt/CaRB#nsubj', [1, 'This']], [2, 'involves'], [3, 'http://www.ieeta-bit.pt/CaRB#dobj', [3, 'trade']], [4, 'http://www.ieeta-bit.pt/CaRB#dep', [4, ' '], [5, 'http://www.ieeta-bit.pt/CaRB#dobj', [5, 'offs']]], [6, 'http://www.ieeta-bit.pt/CaRB#cc', [6, 'and']], [10, 'http://www.ieeta-bit.pt/CaRB#conj', [8, 'http://www.ieeta-bit.pt/CaRB#nsubj', [7, 'http://www.ieeta-bit.pt/CaRB#punct', [7, '{']], [8, 'it'], [9, 'http://www.ieeta-bit.pt/CaRB#punct', [9, '}']]], [10, 'cuts'], [11, 'http://www.ieeta-bit.pt/CaRB#prep', [11, 'against'], [13, 'http://www.ieeta-bit.pt/CaRB#pobj', [12, 'http://www.ieeta-bit.pt/CaRB#det', [12, 'the']], [13, 'grain'], [14, 'http://www.ieeta-bit.pt/CaRB#prep', [14, 'of'], [20, 'http://www.ieeta-bit.pt/CaRB#pobj', [15, 'http://www.ieeta-bit.pt/CaRB#amod', [15, 'existing']], [16, 'http://www.ieeta-bit.pt/CaRB#nmod', [16, 'consumer'], [17, 'http://www.ieeta-bit.pt/CaRB#cc', [17, 'and']], [19, 'http://www.ieeta-bit.pt/CaRB#conj', [18, 'http://www.ieeta-bit.pt/CaRB#advmod', [18, 'even']], [19, 'provider']]], [20, 'conceptions'], [21, 'http://www.ieeta-bit.pt/CaRB#prep', [21, 'of'], [23, 'http://www.ieeta-bit.pt/CaRB#pcomp', [22, 'http://www.ieeta-bit.pt/CaRB#nsubj', [22, 'what']], [23, 'is'], [24, 'http://www.ieeta-bit.pt/CaRB#punct', [24, '`']], [25, 'http://www.ieeta-bit.pt/CaRB#acomp', [25, 'necessary']]]]]]]], [26, 'http://www.ieeta-bit.pt/CaRB#punct', [26, '.']], [27, 'http://www.ieeta-bit.pt/CaRB#punct', [27, 'APOS']], [28, 'http://www.ieeta-bit.pt/CaRB#punct', [28, 'APOSAPOS']]]]]



def get_next_expansion(graph_as_list: list, root_node_id: int, parent_node_id: int, nodes_to_ignore: list) -> list:
    this_w_id, _, *subgraph = graph_as_list
    if not subgraph or this_w_id in nodes_to_ignore:
        return None

    result = [root_node_id, parent_node_id, this_w_id]
    x = [get_next_expansion(sg, root_node_id, this_w_id, nodes_to_ignore) for sg in subgraph]
    x.remove(None)
    return result + x


def get_full_expansion(graph_as_list: list, id_to_expand: int, nodes_to_ignore: list) -> list:
    this_w_id, _, *subgraph = graph_as_list
    if this_w_id == id_to_expand:
        x = [get_next_expansion(sg, this_w_id, this_w_id, nodes_to_ignore) for sg in subgraph]
        if None in x:
            x.remove(None)
        return [[this_w_id, this_w_id, this_w_id], x]
    result = None
    for sg in subgraph:
        rsult = get_full_expansion(sg, id_to_expand, nodes_to_ignore)
        if rsult is not None:
            result = rsult
    return result


def evaluate_path(path, no_outer, root, prev_expansion):
    if path[1] == root:
        no_outer.append([root])
        no_outer[-1].append(path[2])
    # Se o ultimo elemento da expansao for o pai entao basta-me adicionar a frente
    elif path[1] == prev_expansion[-1]:
        no_outer[-1].append(path[2])
    # Se o ultimo elemento NAO for o pai mas o pai esta presente na expansao anterior
    # Estamos a tratar de um ramo e temos de copiar a expansao ate ao pai e completar esse ramo
    elif path[1] in prev_expansion:
        no_outer.append([])
        for node in no_outer[-2]:
            no_outer[-1].append(node)
            if node == path[1]:
                break
        no_outer[-1].append(path[2])
    # Se nao for nenhuma das opcoes e porque vamos completar uma expansao e começar de novo
    else:
        no_outer[-1].append(path[2])
        no_outer.append([root])
    return no_outer

def expand_root(paths, no_outer, root, prev_expansion):
    #print(paths, len(paths))
    for path in paths:
        if isinstance(path, list):
            evaluate = any(isinstance(i, list) for i in path)
            if evaluate:
                if isinstance(path[0], list):
                    no_outer = evaluate_path(path[0], no_outer, root, prev_expansion)
                    if len(path[0]) == 3:
                        expand_root(path[1:], no_outer, root, no_outer[-1])
                    else:
                        expand_root(path[0][1:], no_outer, root, no_outer[-1])
                else:
                    no_outer = evaluate_path(path, no_outer, root, prev_expansion)
                    expand_root(path[2:], no_outer, root, no_outer[-1])
            else:
                if path:
                    no_outer = evaluate_path(path, no_outer, root, prev_expansion)
    return no_outer

# pprint(graph3)
# id_root2xpand = 6
# nodes_to_ignore = []
# # print(get_most_frequent_root(graph, word_ids))
# expansion = get_full_expansion(graph3, id_root2xpand, nodes_to_ignore)
# pprint(expansion)
# pairs = expand_root(expansion[1], [], id_root2xpand, [])
# print(pairs)


import regex as re
from SPARQLWrapper import SPARQLWrapper, POST
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF
from GraphFetch import FetchGraph


def build_insert_query(s, p, o) -> str:
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
        prefix_set.add("PREFIX hlt:<http://www.ieeta-bit.pt/OpenIE_s2_en_v2#>")
        triple.append("hlt:" + uri_comps.group(2))

    query = "\n".join(prefix_set) + "\nINSERT DATA {GRAPH <http://www.ieeta-bit.pt/OpenIE_s2_en_v2#> {" + " ".join(triple) + "}}"
    # print(query)
    return query


def insert_data(query, sparql):
    sparql.setQuery(query)
    for i in range(0, 10):
        try:
            results = sparql.query()
            str_error = None
        except:
            str_error = "Error"
            pass

        if str_error:
            if i == 0:
                print(query)
            print("Error occurred. Attempting to upload triple again.")
            print("Attempt number: ", i)
            sleep(2 * i)  # wait for 2*attempt number seconds before trying to fetch the data again
        else:
            break

connection_uri = "http://hlt.ieeta.pt:8890/sparql"
base_uri = "http://www.ieeta-bit.pt/OpenIE_s2_en_v2#"
graph_name = "http://www.ieeta-bit.pt/OpenIE_s2_en_v2#"
ntx = FetchGraph(base_uri, graph_name, connection_uri)
ids_to_fetch = [*range(14900, 16000, 1)]
previous_id = 14900
nextSentence_uri = URIRef(graph_name+"nextSentence")
sentence_uri = graph_name+"Sentence_0_"
sparql = SPARQLWrapper(connection_uri)
sparql.setMethod(POST)
for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
    idx = ids_to_fetch[i]
    graph = ntx.fetch_graph([idx])
    for s, p, o in graph.triples((None, nextSentence_uri, None)):
        if idx - previous_id > 1:
            print(idx, previous_id)
            from_s = URIRef(sentence_uri + str(previous_id+1))
            to_s = URIRef(sentence_uri + str(previous_id+2))
            insert_data(build_insert_query(from_s, nextSentence_uri, to_s), sparql)
        previous_id = idx