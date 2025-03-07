import pickle
import torch
import pandas as pd
import yaml
import numpy as np
from pprint import pprint
from Helpers import *
from Query_Builder import QueryBuilder
from rdflib import Graph
import ollama
import json
from FetchData import get_graph
import xlsxwriter as xls
from Prompts.OIE_Prompt import OIE_EXTRACTION_PROMPT, TRIPLE_FILTER_PROMPT
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

def format_filter_prompt(all_triples, sentence):
    prompt = TRIPLE_FILTER_PROMPT.format(input_text=sentence,
                                         subject_delimiter="<subject>",
                                         object_delimiter="<object>", predicate_delimiter="<predicate>",
                                         completion_delimiter="<end>",
                                         record_delimiter="<nextTriple>", certainty_delimiter="<rating>")
    for i in range(0, len(all_triples)):
        prompt = prompt + f"\nTriple {i}: {all_triples[i][0]} / {all_triples[i][1]} / {all_triples[i][2]}"
    return prompt

def build_prompt(candidates, sentence, sheet, row, col):
    a1_candidates, a1_sheet = build_candidates("A1", candidates)
    r_candidates, r_sheet = build_candidates("R", candidates)
    a2_candidates, a2_sheet = build_candidates("A2", candidates)
    sheet.write(row, col + 1, a1_sheet)
    sheet.write(row, col + 2, r_sheet)
    sheet.write(row, col + 3, a2_sheet)
    query = OIE_EXTRACTION_PROMPT.format(subject_candidates=a1_candidates, predicate_candidates=r_candidates,
                                         object_candidates=a2_candidates, input_text=sentence, subject_delimiter="<subject>",
                                         object_delimiter="<object>", predicate_delimiter="<predicate>", completion_delimiter="<end>",
                                         record_delimiter="<nextTriple>", certainty_delimiter="<rating>",)

    return query


def extract_from_response(response, sheet, row, column, write_to_excel = False):
    pattern_subject = r"<subject>(.*?)<predicate>"
    pattern_predicate = r"<predicate>(.*?)<object>"
    pattern_object = r"<object>(.*?)<rating>"
    subject_list = re.findall(pattern_subject, response)
    predicate_list = re.findall(pattern_predicate, response)
    object_list = re.findall(pattern_object, response)
    counter = 1
    if len(subject_list) != len(predicate_list) or len(subject_list) != len(object_list) or len(predicate_list) != len(object_list):
        return None, None, None
    else:
        if write_to_excel:
            for i in range(0, len(subject_list)):
                sheet.write(row, column + counter, subject_list[i])
                sheet.write(row, column + counter + 1, predicate_list[i])
                sheet.write(row, column + counter + 2, object_list[i])
                counter += 1
                column += 3
        return subject_list, predicate_list, object_list



def chat(message_with_context):
    stream = ollama.chat(
        #Posso mudar aqui os modelos para o que quiser que ele funciona
        model='qwen2.5:14b-instruct-q5_K_M',
        messages=message_with_context,
    )
    return stream


def query(content):
    # Posso mudar aqui os modelos para o que quiser que ele funciona
    response = ollama.generate(model='qwen2.5:14b-instruct-q5_K_M', prompt=content)
    return response['response']


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


def add_root_predictions(predictions, data):
    # new_data = data.to('cpu')
    new_data = data.clone()
    node_data = new_data.x_dict
    predictions = [[x] for x in predictions]
    nodes_predictions = torch.from_numpy(np.array(predictions)).to(torch.float)
    nodes_tensor = torch.cat((node_data['word'], nodes_predictions), 1)
    new_data['word'].x = nodes_tensor
    return new_data


def generate_word_class(subobj_list, pred_list, word_index_test, targets_test, targets):
    final_list = {}
    for i in range(0, len(subobj_list)):
        if subobj_list[i] != 0 or pred_list[i] != 0:
            if subobj_list[i] != 0:
                final_list[word_index_test[i]] = targets_test.iloc[subobj_list[i]]['originalId']
            else:
                final_list[word_index_test[i]] = targets.iloc[pred_list[i]]['originalId']
        else:
            final_list[word_index_test[i]] = "No"
    return final_list


@torch.no_grad()
def predict(config, device, data, model):
    # if type=='pred':
    #     model = torch.load(config['model_file_pred'])
    # elif type=='subjobj':
    #     model = torch.load(config['model_file_subjobj'])

    # model = model.to(device)
    data = data.to(device)
    out = model(data.x_dict, data.edge_index_dict)
    predictions = out['word'].argmax(dim=1)  # out['word'] is the percentages
    predictions = predictions.cpu().tolist()

    return predictions


def fetch_data(config_data, type_prediction):
    if config_data['load_data_test']:
        if type_prediction == 'pred':
            data = torch.load(config_data['test_data_file_pred'])
            targets = pd.read_pickle(config_data['test_targets_file_pred'])
            with open(config_data['test_indexes_file_pred'], "rb") as f:
                word_index = pickle.load(f)
            return data, targets, word_index
        elif type_prediction == 'subjobj':
            data = torch.load(config_data['test_data_file_subjobj'])
            targets = pd.read_pickle(config_data['test_targets_file_subjobj'])
            with open(config_data['test_indexes_file_subjobj'], "rb") as f:
                word_index = pickle.load(f)
            return data, targets, word_index


def rebuild_graph(qb, sentence_id, config):
    g = Graph()
    doc_id = "0"
    g = build_subgraph(g, qb.build_query_by_sentence_id(doc_id, sentence_id), config['connection_uri'])
    return g


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
data_pred, targets_pred, word_index_pred = fetch_data(config, 'pred')
_, targets_subobj, _ = fetch_data(config, 'subjobj')

#Load up the models onto GPU
pred_model = torch.load(config['model_file_pred'])
pred_model.to(device)
subobj_model = torch.load(config['model_file_subjobj'])
subobj_model.to(device)
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
        history = {'history': []}
        all_triples_extracted = []
        data_single, targets_single, mapped_uris_single, word_index_single = get_graph([int(fetch_idx)], graph_fetch, test = 'Test',
                                                                   targets_test = targets_subobj, embedding=True)

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
                subobj_prediction_list = predict(config, device, new_data, subobj_model)
                word_classification = generate_word_class(subobj_prediction_list, pred_list, word_index_single, targets_single, targets_pred)

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

                tags_dict = {}
                roots_triple_dict = {'A1': '', 'R': '', 'A2': ''}
                for tag_idx in range(0, len(tags_list)):
                    if tags_list[tag_idx] != "No":
                        expansion = get_full_expansion(graph, correct_indexes[tag_idx], [])
                        # Caso tenha expansao, se nao tiver e porque e so ele proprio
                        full_list = expand_root(expansion[1], [], correct_indexes[tag_idx], [])
                        print(full_list)
                        full_list = [[correct_indexes[tag_idx]]] + full_list
                        if tags_list[tag_idx] in tags_dict:
                            tags_dict[tags_list[tag_idx]] = tags_dict[tags_list[tag_idx]] + full_list
                        else:
                            tags_dict[tags_list[tag_idx]] = full_list

                        # Save root to  write excel
                        roots_triple_dict[tags_list[tag_idx]] = roots_triple_dict[tags_list[tag_idx]] + " / " + words_dict[correct_indexes[tag_idx]]

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

                for k2, v2 in tags_dict.items():
                    candidates_list = []
                    for i in range(0, len(v2)):
                        v2[i] = sorted(v2[i], key=lambda x: x)
                    for candidate in v2:
                        sent_mash = ""
                        for word_id in candidate:
                            sent_mash = sent_mash + words_dict[word_id] + " "
                        candidates_list.append(sent_mash)
                        tags_dict[k2] = candidates_list

                # Guarantee that i'm only getting full triples
                if len(tags_dict.keys()) == 3:
                    prompt = build_prompt(tags_dict, sentence, worksheet, row_results+3, col_results)
                    # print(prompt)
                    history['history'].append({'role': 'user', 'content': prompt})
                    chat_message = chat(history['history'])
                    # response = chat_message['message']['content']
                    history['history'].append(chat_message['message'])
                    subjects, predicates, objects = extract_from_response(chat_message['message']['content'], worksheet, row_results+4, col_results)
                    if subjects != None:
                        for i in range(0, len(subjects)):
                            all_triples_extracted.append([subjects[i], predicates[i], objects[i]])
                    # print(response)

        filter_prompt = format_filter_prompt(all_triples_extracted, sentence)
        response = query(filter_prompt)
        subjects, predicates, objects = extract_from_response(response, worksheet,
                                                              row_results + 4, col_results, write_to_excel=True)
        last_idx += len(data_single['word'].x)
        prev_word_id = fetch_idx
        row_results += 6
        col_results = 0
        if row_results > 2000:
            break

workbook.close()
