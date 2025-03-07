from rdflib import Graph
from Helpers import *
from Query_Builder import QueryBuilder
from tqdm import tqdm
from statistics import mode
import xlsxwriter as xls


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


def rebuild_graph(qb, sentence_id, query_url):
    g = Graph()
    doc_id = 0
    g = build_subgraph(g, qb.build_query_by_sentence_id(doc_id, sentence_id), query_url)
    return g


def get_root_node(graph_as_list: list, w_id: int, id_list: list) -> int:
    _NOT_FOUND_ = -1
    this_w_id, _, *subgraph = graph_as_list
    if this_w_id == w_id:
        return this_w_id
    for sg in subgraph:
        current_root = get_root_node(sg, w_id, id_list)
        if current_root != _NOT_FOUND_:
            return this_w_id if this_w_id in id_list else current_root
    return _NOT_FOUND_


def get_most_frequent_root(graph_as_list: list, word_ids: list) -> int:
    return mode([get_root_node(graph_as_list, int(w_id), word_ids) for w_id in word_ids])


def find_central_relationship(relations, graph):
    first_found = {}
    for relation in range(0, len(relations)):
        word_ids = []
        triple_type = ""
        for numbers in relations[relation]:
            word_ids.append(int(numbers[0].split("_")[-1]))
            triple_type = numbers[1]
        word_index = get_most_frequent_root(graph, word_ids)
        # first_found[triple_type] = relations[relation][word_ids.index(word_index)][0]
        first_found[triple_type] = word_index
    return first_found


def fetch_roots(base_uri, g):
    senttext_uri = URIRef(base_uri + "senttext")
    depgraph_uri = URIRef(base_uri + "depGraph")
    id_uri = URIRef(base_uri + "id")
    tags = URIRef(base_uri + "OIE_tags")
    relations = []
    graph = False
    for s, p, o in g.triples((None, senttext_uri, None)):
        graph = list_conll_subgraph(graph=g, root_node=s, transverse_by=depgraph_uri, order_by=id_uri,
                                    main_uri=base_uri)
    if graph:
        if len(graph) == 3:
            # Fetch the words that make up A1, R, A2
            a1 = []
            r = []
            a2 = []
            for s, p, o in g.triples((None, tags, None)):
                word_id = int(s.__str__().split("_")[-1])
                if o.__str__() == "A1":
                    a1.append([s.__str__(), "A1"])
                elif o.__str__() == "A2":
                    a2.append([s.__str__(), "A2"])
                elif o.__str__() == "R":
                    r.append([s.__str__(), "R"])
            if a1 and r and a2:
                relations = find_central_relationship(relations=[a1, r, a2], graph=graph)
    return relations, graph


def fetch_feature(graph, word_uri, base_uri, feature_name):
    for s, p, o in graph.triples((URIRef(word_uri), URIRef(base_uri + feature_name), None)):
        return o.__str__().split("#")[-1]
    return "None"

def fetch_graph_features(graph, word_uri, base_uri):
    edge = fetch_feature(graph, word_uri, base_uri, "hasEdge")
    pos = fetch_feature(graph, word_uri, base_uri,"hasPos")
    posCoarse = fetch_feature(graph, word_uri, base_uri,"hasPoscoarse")

    return [edge, pos, posCoarse]

def export_to_excel(expansion_type, correct, full_list, graph, base_uri, sentence_id, worksheet, row_results, col_results):
    first_run = True
    ids_added = []
    n_jumps_root = 0
    for current_expansion in full_list:
        if first_run:
            current_root = current_expansion[0]
            word_uri = f"{base_uri}word_0_{sentence_id}_{current_root}"
            features = fetch_graph_features(graph, word_uri)
            worksheet.write(row_results, col_results, current_expansion[0])
            worksheet.write(row_results, col_results+1, current_expansion[0])
            worksheet.write(row_results, col_results+2, current_expansion[0])
            col_results = 3
            for feat in features:
                worksheet.write(row_results, col_results, feat)
                col_results += 1
            #Distancia Root - Current
            worksheet.write(row_results, col_results, 0)
            #Direcao Root - Current
            worksheet.write(row_results, col_results+1, 0)
            #Direcao Root - Previous
            worksheet.write(row_results, col_results+2, 0)
            #Direcao Previous - Current
            worksheet.write(row_results, col_results+3, 0)
            worksheet.write(row_results, col_results+4, expansion_type)
            worksheet.write(row_results, col_results+5, 1)

            row_results += 1
            col_results = 0
            first_run = False
            ids_added.append(current_root)
        else:
            prvs_idx = current_root
            for expansion in current_expansion:
                if expansion not in ids_added:
                    ids_added.append(expansion)
                    word_uri = f"{base_uri}word_0_{sentence_id}_{expansion}"
                    features = fetch_graph_features(graph, word_uri)
                    if expansion in correct:
                        worksheet.write(row_results, col_results, current_root)
                        worksheet.write(row_results, col_results + 1, prvs_idx)
                        worksheet.write(row_results, col_results + 2, expansion)
                        col_results = 3
                        for feat in features:
                            worksheet.write(row_results, col_results, feat)
                            col_results += 1

                        # Distancia Root - Current
                        worksheet.write(row_results, col_results, n_jumps_root)

                        # Direcao Root - Current
                        if expansion < current_root:
                            worksheet.write(row_results, col_results + 1, -1)
                        elif expansion == current_root:
                            worksheet.write(row_results, col_results + 1, 0)
                        else:
                            worksheet.write(row_results, col_results + 1, 1)

                        # Direcao Root - Previous
                        if prvs_idx < current_root:
                            worksheet.write(row_results, col_results + 2, -1)
                        elif prvs_idx == current_root:
                            worksheet.write(row_results, col_results + 2, 0)
                        else:
                            worksheet.write(row_results, col_results + 2, 1)

                        # Direcao Previous - Current
                        if expansion < prvs_idx:
                            worksheet.write(row_results, col_results + 3, -1)
                        elif expansion == prvs_idx:
                            worksheet.write(row_results, col_results + 3, 0)
                        else:
                            worksheet.write(row_results, col_results + 3, 1)

                        worksheet.write(row_results, col_results + 4, expansion_type)
                        worksheet.write(row_results, col_results + 5, 1)
                    else:
                        worksheet.write(row_results, col_results, current_root)
                        worksheet.write(row_results, col_results + 1, prvs_idx)
                        worksheet.write(row_results, col_results + 2, expansion)
                        col_results = 3
                        for feat in features:
                            worksheet.write(row_results, col_results, feat)
                            col_results += 1
                        # Distancia Root - Current
                        worksheet.write(row_results, col_results, n_jumps_root)

                        # Direcao Root - Current
                        if expansion < current_root:
                            worksheet.write(row_results, col_results + 1, -1)
                        elif expansion == current_root:
                            worksheet.write(row_results, col_results + 1, 0)
                        else:
                            worksheet.write(row_results, col_results + 1, 1)

                        # Direcao Root - Previous
                        if prvs_idx < current_root:
                            worksheet.write(row_results, col_results + 2, -1)
                        elif prvs_idx == current_root:
                            worksheet.write(row_results, col_results + 2, 0)
                        else:
                            worksheet.write(row_results, col_results + 2, 1)

                        # Direcao Previous - Current
                        if expansion < prvs_idx:
                            worksheet.write(row_results, col_results + 3, -1)
                        elif expansion == prvs_idx:
                            worksheet.write(row_results, col_results + 3, 0)
                        else:
                            worksheet.write(row_results, col_results + 3, 1)

                        worksheet.write(row_results, col_results + 4, expansion_type)
                        worksheet.write(row_results, col_results + 5, 0)
                    prvs_idx = expansion
                    row_results += 1
                    col_results = 0
                n_jumps_root += 1
                prvs_idx = expansion
            n_jumps_root = 0
    return row_results, col_results


query_url = "http://hlt.ieeta.pt:8890/sparql"
base_uri = "http://www.ieeta-bit.pt/OpenIE_s2_en_v2#"
graph_name = "http://www.ieeta-bit.pt/OpenIE_s2_en_v2#"
qb = QueryBuilder(base_uri, graph_name)

workbook = xls.Workbook("/home/grsilva/GraphBuilderAPI_v2/ML_Only/Expansion_Dataset/Expansions_Fixed_Teste.xlsx")
worksheet = workbook.add_worksheet()
rows = 0
cols = 0
worksheet.write(rows, cols+0, "Base Root")
worksheet.write(rows, cols+1, "Previous Root")
worksheet.write(rows, cols+2, "Current Id")
worksheet.write(rows, cols+3, "Edge")
worksheet.write(rows, cols+4, "Pos")
worksheet.write(rows, cols+5, "PosCoarse")
worksheet.write(rows, cols+6, "D_Base_Current")
worksheet.write(rows, cols+7, "P_Base_Previous")
worksheet.write(rows, cols+8, "P_Base_Current")
worksheet.write(rows, cols+9, "P_Previous_Current")
worksheet.write(rows, cols+10, "ExpType")
worksheet.write(rows, cols+11, "Target")

rows += 1
ids_to_fetch = [*range(5001, 6000, 1)]
#Query sentence by sentence
for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
    idx = ids_to_fetch[i]
    g = rebuild_graph(qb, idx, query_url)
    # Fetch the roots
    roots, graph = fetch_roots(base_uri, g)
    # Fetch the correct expansion ids
    if roots:
        correct_expansions = {'A1': [], 'R': [], 'A2': []}
        for s, p, o in g.triples((None, URIRef(base_uri + "OIE_tags"), None)):
            if o.__str__() != "No":
                word_id = int(s.__str__().split("_")[-1])
                correct_expansions[o.__str__()].append(int(word_id))
        #Fetch the expanded expansions starting from the root
        for k, v in roots.items():
            expansion = get_full_expansion(graph, v, [])
            full_expansion_list = expand_root(expansion[1], [], v, [])
            full_expansion_list = [[v]] + full_expansion_list
            rows, cols = export_to_excel(k, correct_expansions[k], full_expansion_list, g, graph_name, idx, worksheet, rows, cols)


#Save dataset
workbook.close()