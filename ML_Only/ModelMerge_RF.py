import itertools
import pickle
import torch
import pandas as pd
import yaml
import numpy as np
from pprint import pprint
from Helpers import *
from Query_Builder import QueryBuilder
from rdflib import Graph
import json
from FetchData import get_graph
import xlsxwriter as xls
import re


def build_candidates(key, candidates):
    candidate_string = f"""Options for {key}: """
    candidate_sheet = ""
    for candidate in range(0, len(candidates[key])):
        candidate_string += f"""{candidates[key][candidate]} / """
        candidate_sheet += f"""{candidates[key][candidate]} / """
    candidate_string = candidate_string[:-3]
    candidate_sheet = candidate_sheet[:-3]
    return candidate_string, candidate_sheet


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
    """
    Expands paths by recursively evaluating and expanding nested lists.

    Parameters:
    paths (list): A list of paths to be expanded.
    no_outer (list): The current state of the outer path being evaluated.
    root (int): The root node ID.
    prev_expansion (list): The previous expansion result.

    Returns:
    list: The updated state of the outer path after expansion.

    Notes:
    - This function recursively evaluates each path and expands nested lists.
    - It handles different structures within paths to ensure proper expansion.
    """
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
    """
    Recursively expands the graph to find all possible paths from a specified node.

    Parameters:
    graph_as_list (list): A nested list representation of the graph.
    root_node_id (int): The ID of the root node.
    parent_node_id (int): The ID of the parent node.
    nodes_to_ignore (list): A list of node IDs to ignore during expansion.

    Returns:
    list: A list representing all paths from the specified node, or None if no path is found.

    Notes:
    - This function assumes that each subgraph in `graph_as_list` has a structure where the first element is the current node ID.
    - The function recursively explores all subgraphs and concatenates them to form complete paths.
    """

    this_w_id, _, *subgraph = graph_as_list
    if not subgraph or this_w_id in nodes_to_ignore:
        return None

    result = [root_node_id, parent_node_id, this_w_id]
    x = [get_next_expansion(sg, root_node_id, this_w_id, nodes_to_ignore) for sg in subgraph]
    x.remove(None)
    return result + x


def get_full_expansion(graph_as_list: list, id_to_expand: int, nodes_to_ignore: list) -> list:
    """
    Recursively expands a graph to find all paths from a specified node.

    Parameters:
    graph_as_list (list): A nested list representation of the graph.
    id_to_expand (int): The ID of the node to expand.
    nodes_to_ignore (list): A list of node IDs to ignore during expansion.

    Returns:
    list: A list representing all paths from the specified node, or None if no path is found.

    Notes:
    - This function assumes that each subgraph in `graph_as_list` has a structure where the first element is the current node ID.
    - The function recursively explores all subgraphs and concatenates them to form complete paths.
    """
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


def add_root_predictions(predictions, data):
    """
    Adds root predictions to the node features of a graph.

    Parameters:
    predictions (list of float): A list of predicted values.
    data (torch_geometric.data.Data): The input graph data object.

    Returns:
    torch_geometric.data.Data: The modified graph data object with added predictions.

    Notes:
    - This function assumes that the input `data` has a node feature dictionary named 'word'.
    - The predictions are converted to a tensor and concatenated with the existing 'word' features.
    """
    new_data = data.clone()
    node_data = new_data.x_dict
    predictions = [[x] for x in predictions]
    nodes_predictions = torch.from_numpy(np.array(predictions)).to(torch.float)
    nodes_tensor = torch.cat((node_data['word'], nodes_predictions), 1)
    new_data['word'].x = nodes_tensor
    return new_data


def generate_word_class(sub_list, pred_list, obj_list, word_index_test, targets_sub, targets_pred, targets_obj):
    final_list = {}
    for i in range(0, len(sub_list)):
        if sub_list[i] != 0:
            final_list[word_index_test[i]] = targets_sub.iloc[sub_list[i]]['originalId']
        elif obj_list[i] != 0:
            final_list[word_index_test[i]] = targets_obj.iloc[obj_list[i]]['originalId']
        elif pred_list[i] != 0:
            final_list[word_index_test[i]] = targets_pred.iloc[pred_list[i]]['originalId']
        else:
            final_list[word_index_test[i]] = "No"
    return final_list


@torch.no_grad()
def predict(config, device, data, model):
    data = data.to(device)
    out = model(data.x_dict, data.edge_index_dict)
    predictions = out['word'].argmax(dim=1)  # out['word'] is the percentages
    predictions = predictions.cpu().tolist()

    return predictions


def fetch_data(test_data_file, target_file, indexes_file):
    data = torch.load(test_data_file)
    targets = pd.read_pickle(target_file)
    with open(indexes_file, "rb") as f:
        word_index = pickle.load(f)
    return data, targets, word_index


def rebuild_graph(qb, sentence_id, config):
    g = Graph()
    doc_id = "0"
    g = build_subgraph(g, qb.build_query_by_sentence_id(doc_id, sentence_id), config['connection_uri'])
    return g


def fetch_feature(graph, word_uri, base_uri, feature_name):
    for s, p, o in graph.triples((URIRef(word_uri), URIRef(base_uri + feature_name), None)):
        return o.__str__().split("#")[-1]
    return "None"


def fetch_graph_features(graph, word_uri, base_uri):
    edge = fetch_feature(graph, word_uri, base_uri, "hasEdge")
    pos = fetch_feature(graph, word_uri, base_uri,"hasPos")
    posCoarse = fetch_feature(graph, word_uri, base_uri,"hasPoscoarse")

    return [edge, pos, posCoarse]


def build_features(full_list, graph, base_uri, sentence_id):
    n_jumps_root = 0
    base_root = full_list[0][0]
    feature_list = []
    for current_expansion in full_list:
        prvs_idx = base_root
        feature_list.append([])
        for expansion in current_expansion:
            feature_list[len(feature_list) - 1].append([])
            word_uri = f"{base_uri}word_0_{sentence_id}_{expansion}"
            features = fetch_graph_features(graph, word_uri, base_uri)
            feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(base_root)
            feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(prvs_idx)
            feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(expansion)
            for feat in features:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(feat)
            # Distancia Root - Current
            feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(n_jumps_root)

            # Direcao Root - Current
            if expansion < base_root:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(-1)
            elif expansion == base_root:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(0)
            else:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(1)

            # Direcao Root - Previous
            if prvs_idx < base_root:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(-1)
            elif prvs_idx == base_root:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(0)
            else:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(1)

            # Direcao Previous - Current
            if expansion < prvs_idx:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(-1)
            elif expansion == prvs_idx:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(0)
            else:
                feature_list[len(feature_list) - 1][len(feature_list[len(feature_list) - 1])-1].append(1)

            prvs_idx = expansion
            n_jumps_root += 1
            prvs_idx = expansion
        n_jumps_root = 0
    return feature_list


def predict_expansion(model, dataset, encoders, scores):
    proba_expansions = []
    for expansion in dataset:
        # predicted_expansions.append([])
        x_predict = pd.DataFrame(expansion, columns=["Base Root", "Previous Root", "Current Id", "Edge", "Pos", "PosCoarse", "D_Base_Current", "P_Base_Previous", "P_Base_Current", "P_Previous_Current"])
        x_predict['Edge'] = encoders['edge'].transform(x_predict['Edge'])
        x_predict['Pos'] = encoders['pos'].transform(x_predict['Pos'])
        x_predict['PosCoarse'] = encoders['poscoarse'].transform(x_predict['PosCoarse'])
        # y_pred = model.predict(x_predict)
        y_prob = model.predict_proba(x_predict)
        #print(y_prob)
        # print(len(y_prob))
        if len(y_prob) != 1:
            # avg_positive = 0
            # for i in range(1, len(y_prob)):
            #     avg_positive += y_prob[i][1]
            # avg_positive = avg_positive / (len(y_prob)-1)
            # proba_expansions.append(avg_positive)

            #Quando der negativo ir ver se tem algum elemento em comum. Fazer a media desses dois conjuntos para ver se
            #é positiva ou não. Caso seja aceito os dois conjuntos. Continuar até não haver elementos em comum ou dar negativo.
            #Caso sejam tod  os negativos ficar apenas com o root.
            for i in range(0, len(y_prob)):
                scores[x_predict['Current Id'][i]] = y_prob[i][1]
        else:
            scores[x_predict['Current Id'][0]] = 0
            #proba_expansions.append(0)
        # for i in range(0, len(y_pred)):
        #     if y_pred[i] == 1:
        #         predicted_expansions[len(predicted_expansions)-1].append(expansion[i][2])
    # return proba_expansions
    return scores


def filter_expansions(expansions, word_indexes):
    try:
        idx_r = word_indexes[expansions.index('R')]
    except:
        return False
    for i in range(0, len(expansions)):
        if expansions[i] == 'A1':
            if word_indexes[i] > idx_r:
                expansions[i] = 'No'
        elif expansions[i] == 'A2':
            if word_indexes[i] < idx_r:
                expansions[i] = 'No'
    return expansions


def fetch_valid_triples(triples, scores):
    valid_triples = {'A1': '', 'R': '', 'A2': ''}
    #Fetch the R first.
    valid_triples['R'] = triples['R']
    #Filter the A1 and A2
    valid_triples['A1'] = validity_filter(triples['A1'], scores['A1'], valid_triples['R'])
    valid_triples['A2'] = validity_filter(triples['A2'], scores['A2'], valid_triples['R'])
    return valid_triples


def validity_filter(triples, scores, r_triple):
    max_indexes = [i[0] for i in sorted(enumerate(scores), key=lambda k: k[1], reverse=True)]
    found = True
    for idx in max_indexes:
        for triple_idx in triples[idx]:
            if triple_idx in r_triple:
                found = False
                break

        if found:
            return triples[idx]
        else:
            found = True


def get_acumuladas(expansion, root):
    acumulada = 0
    current_it = 0
    acumulada_list_maior = []
    nums_considered = []
    block_num_maior = {}
    current_block = 0
    total_num = 0
    # dicionario para contar o numero de blocos
    if expansion:
        for inner_list in expansion:
            current_block += 1
            block_num_maior[current_block] = -1
            if len(inner_list) > 1:
                for num in range(0, len(inner_list)):
                    if inner_list[num] not in nums_considered and inner_list[num] != root:
                        nums_considered.append(inner_list[num])
                        acumulada += scores[inner_list[num]]
                        current_it += 1
                        acumulada_list_maior.append(acumulada / (current_it))
                    else:
                        acumulada_list_maior.append(-1)
                total_num = total_num + len(inner_list)
                block_num_maior[current_block] = total_num
            else:
                acumulada_list_maior.append(-2)
    else:
        acumulada_list_maior.append(-1)
    return acumulada_list_maior, block_num_maior


def sao_menores(lista:list, elemento: int) -> list:
    return True if all(elemento >= x for x in lista) else False


def load_pickle(file_name):
    pkl_file = open(file_name, 'rb')
    loaded_file = pickle.load(pkl_file)
    return loaded_file


def find_expansion(acumulada_list, blocks_dict, full_list):
    current_max = min(acumulada_list, key=lambda x: abs(x - 0.5))
    max_arg = acumulada_list.index(current_max)
    # current_max = max(acumulada_list)
    # max_arg = np.argmax(acumulada_list)
    expansion = []
    for i in range(0, len(full_list)):
        if max_arg < blocks_dict[i+1]:
            # print("HERE", full_list[i])
            expansion = expansion + full_list[i]
            break
        else:
            expansion = expansion + full_list[i]
    # print(expansion)
    expansion = list(set(expansion))
    expansion.sort()
    return expansion, current_max

merge_config = open('/home/grsilva/GraphBuilderAPI_v2/ML_Only/Configs/Merge_Conf.yaml', 'r')
graph_fetch_config = open('/home/grsilva/GraphBuilderAPI_v2/ML_Only/Configs/GraphML_OpenIE_EN_Teste.yaml', 'r')

config = yaml.load(merge_config, Loader=yaml.FullLoader)
graph_fetch = yaml.load(graph_fetch_config, Loader=yaml.FullLoader)
base_uri = config['base_uri']

# Opening LLM history for later
# with open('QueryHistory/history.json') as f:
#     history = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qb = QueryBuilder(config['base_uri'], config['graph_name'])
# Fetch the data to test.
if config['load_data_test']:
    data_pred, targets_pred, word_index_pred = fetch_data(config['test_data_file_pred'], config['test_targets_file_pred'], config['test_indexes_file_pred'])
    _, targets_sub, _ = fetch_data(config['test_data_file_sub'], config['test_targets_file_sub'], config['test_indexes_file_sub'])
    _, targets_obj, _ = fetch_data(config['test_data_file_obj'], config['test_targets_file_obj'], config['test_indexes_file_obj'])

#Load up the models onto GPU
pred_model = torch.load(config['model_file_pred'])
pred_model.to(device)
sub_model = torch.load(config['model_file_sub'])
sub_model.to(device)
obj_model = torch.load(config['model_file_obj'])
obj_model.to(device)

#Load Model and label encoders
expansion_model = load_pickle(config['expansion_model_path'])
encoder_1 = load_pickle(config['edge_encoder'])
encoders = {'edge': load_pickle(config['edge_encoder']), 'pos': load_pickle(config['pos_encoder']),
            'poscoarse': load_pickle(config['poscoarse_encoder'])}

le_name_mapping = dict(zip(encoders['poscoarse'].classes_, encoders['poscoarse'].transform(encoders['poscoarse'].classes_)))

# Predict the predicates first
prediction_list = predict(config, device, data_pred, pred_model)

# Free up GPU memory
del pred_model
del data_pred

# Results Workbook
workbook = xls.Workbook(config['excel_save_path'])
worksheet = workbook.add_worksheet()
row_results = 0
col_results = 0
prev_word_id = 0
last_idx = 0

for word_idx in word_index_pred:
    fetch_idx = word_idx.split("_")[-2]
    if fetch_idx != prev_word_id:
        print(fetch_idx)
        all_triples_extracted = []
        data_single, targets_single, mapped_uris_single, word_index_single = get_graph([int(fetch_idx)], graph_fetch, test = 'Test',
                                                                   targets_test = targets_sub, embedding=True)

        worksheet.write(row_results, col_results, "Sentence")
        worksheet.write(row_results + 1, col_results, "Roots Found")
        worksheet.write(row_results + 2, col_results, "Real Triple")
        worksheet.write(row_results + 3, col_results, "Triple Options")
        worksheet.write(row_results + 4, col_results, "Predicted Triple")
        for pred_idx in range(0, len(data_single['word'].x)):
            if prediction_list[last_idx+pred_idx] == 1:
                pred_list = [0]*len(data_single['word'].x)
                pred_list[pred_idx] = 1


                new_data = add_root_predictions(pred_list, data_single)
                # new_data = new_data.to(device)
                sub_prediction_list = predict(config, device, new_data, sub_model)
                obj_prediction_list = predict(config, device, new_data, obj_model)
                # print("Sub", sub_prediction_list)
                # print("Obj", obj_prediction_list)
                # List merge, fica com o maior resultado.
                subobj_prediction_list = [max(sub_prediction_list[i], obj_prediction_list[i]) for i in range(len(sub_prediction_list))]
                word_classification = generate_word_class(sub_prediction_list, pred_list, obj_prediction_list, word_index_single, targets_sub, targets_pred, targets_obj)
                # print(word_classification)
                # Query the LLM
                tags_list = list(word_classification.values())
                current_sentence_uri = base_uri + "Sentence_0_" + fetch_idx
                g = rebuild_graph(qb, fetch_idx, config)
                for s, p, o in g.triples((URIRef(current_sentence_uri), URIRef(base_uri + "senttext"), None)):
                    sentence = o.__str__()
                    if col_results == 0:
                        worksheet.write(row_results, col_results+1, sentence)
                        worksheet.write(row_results, col_results+2, fetch_idx)

                    graph = list_conll_subgraph(graph=g, root_node=s,
                                                transverse_by=URIRef(base_uri + "depGraph"),
                                                order_by=URIRef(base_uri + "id"),
                                                main_uri=base_uri)

                correct_indexes = []
                words_dict = {}
                for s, p, o in g.triples((None, URIRef(base_uri + "word"), None)):
                    words_dict[int(s.__str__().split("_")[-1])] = o.__str__()
                    correct_indexes.append(int(s.__str__().split("_")[-1]))

                tags_dict = {'A1': [], 'R': [], 'A2': []}
                tags_dict_scores = {'A1': 0, 'A2': 0, 'R': 0}
                # tags_dict = {'A1': '', 'R': '', 'A2': ''}
                roots_triple_dict = {'A1': '', 'R': '', 'A2': ''}
                graph_features = {}
                tags_list = filter_expansions(tags_list, correct_indexes)
                # print(tags_list)
                if tags_list:
                    for tag_idx in range(0, len(tags_list)):
                        if tags_list[tag_idx] != "No":
                            type_tag = tags_list[tag_idx]
                            # print("Type:", tags_list[tag_idx])
                            word_uri = base_uri + "Word_0_" + fetch_idx + "_" + str(correct_indexes[tag_idx])
                            expansion = get_full_expansion(graph, correct_indexes[tag_idx], [])
                            # Caso tenha expansao, se nao tiver e porque e so ele proprio
                            full_list = expand_root(expansion[1], [], correct_indexes[tag_idx], [])
                            # print("Expansao:", full_list)
                            #Caso nao tenha expansao (ninguem depende desta palavra) nao me vale a pena estar a mandar a RF porque e so a palavra
                            if full_list:
                                feature_list = build_features(full_list, g, base_uri, fetch_idx)
                            else:
                            #     #A palavra esta sozinha
                                full_list = [[correct_indexes[tag_idx]]]
                                feature_list = build_features(full_list, g, base_uri, fetch_idx)

                            scores = {}
                            scores = predict_expansion(expansion_model, feature_list, encoders, scores)
                            root = full_list[0][0]
                            exp_menor = [lst for lst in full_list if sao_menores(lst, root)]
                            exp_maior = [lst for lst in full_list if not sao_menores(lst, root)]

                            exp_menor.sort(reverse=True)
                            # print("MENOR", exp_menor)
                            # print("MAIOR", exp_maior)
                            #Prob acumulada
                            acumulada_list_menor, block_num_menor = get_acumuladas(exp_menor, root)
                            acumulada_list_maior, block_num_maior = get_acumuladas(exp_maior, root)

                            # print("ACUM_MENOR", acumulada_list_menor)
                            # print("ACUM_MAIOR", acumulada_list_maior)
                            #
                            # print("BLOCK_MENOR", block_num_menor, max(acumulada_list_menor))
                            # print("BLOCK_MAIOR", block_num_maior, max(acumulada_list_maior))

                            result_maior = min(acumulada_list_maior, key=lambda x: abs(x - 0.5))
                            result_menor = min(acumulada_list_menor, key=lambda x: abs(x - 0.5))

                            if result_maior > result_menor:
                                expansion, max_score = find_expansion(acumulada_list_maior, block_num_maior, exp_maior)
                            else:
                                expansion, max_score = find_expansion(acumulada_list_menor, block_num_menor, exp_menor)

                            if tags_dict_scores[type_tag] < max_score and max_score > 0.5:
                                tags_dict[type_tag] = expansion
                                tags_dict_scores[type_tag] = max_score
                            else:
                                if tags_dict_scores[type_tag] == 0:
                                    tags_dict[type_tag] = [root]
                                    tags_dict_scores[type_tag] = 0.5

                            # print("TAGS", tags_dict)
                            # for j in range(0, len(proba_expansions)):
                            #     tags_dict_scores[type + '_Score'].append(proba_expansions[j])
                            #     tags_dict[type].append(full_list[j])
                            # if max(proba_expansions) > tags_dict_scores[tags_list[tag_idx] + '_Score']:
                            #     index_max = np.argmax(proba_expansions)
                            #     tags_dict_scores[tags_list[tag_idx] + '_Score'] = proba_expansions[index_max]
                            #     tags_dict[tags_list[tag_idx]] = full_list[index_max].sort()

                            # ---- PREVIOUS WITHOUT PROBABILITY ------
                            # Expansions related to the WORD in question.
                            # predicted_expansions.sort()
                            # full_list = list(k for k, _ in itertools.groupby(predicted_expansions))
                            # print("Predicted:", predicted_expansions)
                            # #Flatten the list and extract the ids we want to for our triple
                            # flat_full_list = []
                            # for first_layer in predicted_expansions:
                            #     for second_layer in first_layer:
                            #         if second_layer not in flat_full_list:
                            #             flat_full_list.append(second_layer)
                            # flat_full_list.sort()
                            # print(flat_full_list)
                            # ids_to_consider = [*range(flat_full_list[0], flat_full_list[-1]+1, 1)]
                            # print("Ids to fetch", ids_to_consider)

                            # if tags_list[tag_idx] in tags_dict:
                            #     # tags_dict[tags_list[tag_idx]] = tags_dict[tags_list[tag_idx]] + full_list
                            #     tags_dict[tags_list[tag_idx]] = tags_dict[tags_list[tag_idx]] + ids_to_consider
                            # else:
                            #     # tags_dict[tags_list[tag_idx]] = full_list
                            #     tags_dict[tags_list[tag_idx]] = ids_to_consider
                            #
                            # # Save root to  write excel
                            # roots_triple_dict[tags_list[tag_idx]] = roots_triple_dict[tags_list[tag_idx]] + " / " + \
                            #                                         words_dict[correct_indexes[tag_idx]]
                    #tags_dict = fetch_valid_triples(tags_dict, tags_dict_scores)
                    # print(tags_dict)
                    incomplete = False
                    for k, v in tags_dict.items():
                        if v == None:
                            incomplete = True

                    if not incomplete:
                        num_loops = 1
                        for k, v in roots_triple_dict.items():
                            worksheet.write(row_results + 1, col_results + num_loops, v)
                            num_loops += 1

                        # Write the correct triple to excel
                        correct_triple_dict = {'A1': '', 'R': '', 'A2': ''}
                        for s, p, o in g.triples((None, URIRef(base_uri + "OIE_tags"), None)):
                            if o.__str__() != "No":
                                word_id = int(s.__str__().split("_")[-1])
                                correct_triple_dict[o.__str__()] = correct_triple_dict[o.__str__()] + f" {words_dict[word_id]}"

                        if col_results == 0:
                            num_loops = 1
                            for k, v in correct_triple_dict.items():
                                worksheet.write(row_results+2, col_results+num_loops, v)
                                num_loops += 1

                        # Fetch the words of each triple
                        for k, v in tags_dict.items():
                            sent = ''
                            for value in v:
                                word = words_dict[value]
                                sent += word + ' '
                            if k == "A1":
                                worksheet.write(row_results + 4, col_results+1, sent)
                            elif k == "R":
                                worksheet.write(row_results + 4, col_results+2, sent)
                            else:
                                worksheet.write(row_results + 4, col_results+3, sent)
                        col_results+=4
                    # for k2, v2 in tags_dict.items():
                    #     candidates_list = []
                    #     for i in range(0, len(v2)):
                    #         v2[i] = sorted(v2[i], key=lambda x: x)
                    #     for candidate in v2:
                    #         sent_mash = ""
                    #         for word_id in candidate:
                    #             sent_mash = sent_mash + words_dict[word_id] + " "
                    #         candidates_list.append(sent_mash)
                    #         tags_dict[k2] = candidates_list

            #         # Guarantee that i'm only getting full triples
            #         if len(tags_dict.keys()) == 3:
            #             prompt = build_prompt(tags_dict, sentence, worksheet, row_results+3, col_results)
            #             # print(prompt)
            #             history['history'].append({'role': 'user', 'content': prompt})
            #             chat_message = chat(history['history'])
            #             # response = chat_message['message']['content']
            #             history['history'].append(chat_message['message'])
            #             subjects, predicates, objects = extract_from_response(chat_message['message']['content'], worksheet, row_results+4, col_results)
            #             if subjects != None:
            #                 for i in range(0, len(subjects)):
            #                     all_triples_extracted.append([subjects[i], predicates[i], objects[i]])
            #             # print(response)
            #
            # filter_prompt = format_filter_prompt(all_triples_extracted, sentence)
            # response = query(filter_prompt)
            # subjects, predicates, objects = extract_from_response(response, worksheet,
            #                                                       row_results + 4, col_results, write_to_excel=True)
        last_idx += len(data_single['word'].x)
        prev_word_id = fetch_idx
        row_results += 6
        col_results = 0
        if row_results > 20:
            break

workbook.close()
