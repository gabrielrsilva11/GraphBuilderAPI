import xlsxwriter as xls
import pandas as pd
from Helpers import *
import yaml


def fetch_graph(qb, sentence_id, connection_string):
    g = Graph()
    doc_id = 0
    g = build_subgraph(g, qb.build_query_by_sentence_id(doc_id, int(sentence_id)), connection_string)
    return g

def fetch_root_word_pair(g, base_uri, root_word_id, valid_paths):
    #print(root_word_id)
    for s, p, o in g.triples((URIRef(root_word_id), URIRef(base_uri+"depGraph"), None)):
        current_path = []
        for s2, p2, o2 in g.triples((o, URIRef(base_uri+"word"), None)):
            word_id = s2.__str__().split("_")[-1]
            current_path.append([o2.__str__(), word_id])
        current_path = fetch_root_word_pair(g, base_uri, o.__str__(), current_path)
        valid_paths.append(current_path)
    return valid_paths


def remove_outer_lists(paths, no_outer):
    #print(paths, len(paths))
    if len(paths) > 1:
        # print("A")
        for path in paths:
            if len(path) > 1:
                #print(path, len(path))
                if type(path[1]) == list:
                    remove_outer_lists(path, no_outer)
                else:
                    no_outer.append(path)
            else:
                no_outer.append(path[0])
    else:
        # print("B")
        # print(paths[0])
        no_outer.append(paths[0])
    return no_outer

def order_word_paths(valid_paths):
    string_paths = []
    if len(valid_paths) == 1:
        return [valid_paths[0][0]]
    else:
        for i in range(1, len(valid_paths)):
            current_path = [valid_paths[0]]
            if len(valid_paths[i]) == 1:
                no_outer = remove_outer_lists(valid_paths[i], [])
                current_path.append(no_outer[0])
            else:
                no_outer = remove_outer_lists(valid_paths[i], [])
                for paths_2 in no_outer:
                    current_path.append(paths_2)
            current_path.sort(key=lambda x: int(x[1]))
            joint_string_list = [''.join(ele[0]) for ele in current_path]
            if len(joint_string_list) > 1:
                string_paths.append(' '.join(joint_string_list))
            else:
                string_paths.append(joint_string_list[0])
    return string_paths

def convert_results(base_uri, graph_name, connection_string):
    base_uri = "http://www.ieeta-bit.pt/OpenIE_s2_pt#"
    graph_name = "http://www.ieeta-bit.pt/OpenIE_s2_pt#"
    connection_string = "http://34.175.171.126:8890/sparql"
    qb = QueryBuilder(base_uri, graph_name)

    results = pd.read_excel('Results/Validation/GridSearch/Resultados_GridSearch_PT_Teste.xlsx')
    workbook = xls.Workbook('Results/Validation/Processed_GridSearch_PT_Teste.xlsx')
    worksheet = workbook.add_worksheet()
    row_excel = 0
    column = 0

    format = workbook.add_format()
    format.set_pattern(1)
    format.set_bg_color('gray')

    worksheet.write(row_excel, column, "Sentence")
    worksheet.write(row_excel+1, column, "A1_Dataset")
    worksheet.write(row_excel+2, column, "A1_Predicted")
    worksheet.write(row_excel+3, column, "R_Dataset")
    worksheet.write(row_excel+4, column, "R_Predicted")
    worksheet.write(row_excel+5, column, "A2_Dataset")
    worksheet.write(row_excel+6, column, "A2_Predicted")
    a1_k = 1
    a2_k = 1
    r_k = 1
    current_sentence = 0
    for index, row in results.iterrows():
        current_sent_id = row['word_id'].split("_")[-2]
        if current_sent_id != current_sentence:
            a1_k = 1
            a2_k = 1
            r_k = 1
            current_sentence = current_sent_id
            current_sentence_uri = base_uri + "Sentence_0_" + current_sentence
            #Fetch the sentence details
            triple_list = []
            g = fetch_graph(qb, current_sentence, connection_string)
            #Fetch sentence text
            worksheet.write(row_excel, column + 1, current_sentence)
            for s, p, o in g.triples((URIRef(current_sentence_uri), URIRef(base_uri+"senttext"), None)):
                graph = list_conll_subgraph(graph=g, root_node=s, transverse_by=URIRef(base_uri + "depGraph"),
                                            order_by=URIRef(base_uri + "id"),
                                            main_uri=base_uri)
                worksheet.write(row_excel, column+2, o.__str__())
                worksheet.write(row_excel, column+4, graph.__str__())

            #Fetch the Dataset labels
            #word = ""
            word_list = []
            for s, p, o in g.triples((None, URIRef(base_uri+"OIE_tags"), Literal("A1"))):
                for s2, p2, o2 in g.triples((s, URIRef(base_uri+"word"), None)):
                    # worksheet.write(row_excel + 1, column+1, o2.__str__())
                    # column+=1
                    word_list.append([o2.__str__() + " ", s2.__str__().split("_")[-1]])
            word_list.sort(key=lambda x: int(x[1]))
            joint_string_list = [''.join(ele[0]) for ele in word_list]
            if len(joint_string_list) > 1:
                worksheet.write(row_excel + 1, column + 1, ''.join(joint_string_list))
            elif word_list:
                worksheet.write(row_excel + 1, column + 1, word_list[0][0])

            column = 0
            #word = ""
            word_list = []
            for s, p, o in g.triples((None, URIRef(base_uri+"OIE_tags"), Literal("R"))):
                for s2, p2, o2 in g.triples((s, URIRef(base_uri+"word"), None)):
                    #word = word + o2.__str__() + " "
                    word_list.append([o2.__str__() + " ", s2.__str__().split("_")[-1]])
                word_list.sort(key=lambda x: int(x[1]))
                joint_string_list = [''.join(ele[0]) for ele in word_list]
                if len(joint_string_list) > 1:
                    worksheet.write(row_excel + 3, column + 1, ''.join(joint_string_list))
                else:
                    worksheet.write(row_excel + 3, column + 1, word_list[0][0])

            column = 0
            word_list = []
            for s, p, o in g.triples((None, URIRef(base_uri+"OIE_tags"), Literal("A2"))):
                for s2, p2, o2 in g.triples((s, URIRef(base_uri+"word"), None)):
                    # worksheet.write(row_excel + 5, column+1, o2.__str__())
                    # column+=1
                    #word = word + o2.__str__() + " "
                    word_list.append([o2.__str__() + " ", s2.__str__().split("_")[-1]])
                word_list.sort(key=lambda x: int(x[1]))
                joint_string_list = [''.join(ele[0]) for ele in word_list]
                if len(joint_string_list) > 1:
                    worksheet.write(row_excel + 5, column + 1, ''.join(joint_string_list))
                else:
                    worksheet.write(row_excel + 5, column + 1, word_list[0][0])

            column = 0
            row_excel+=8
            worksheet.write(row_excel, column, "Sentence")
            worksheet.write(row_excel + 1, column, "A1_Dataset")
            worksheet.write(row_excel + 2, column, "A1_Predicted")
            worksheet.write(row_excel + 3, column, "R_Dataset")
            worksheet.write(row_excel + 4, column, "R_Predicted")
            worksheet.write(row_excel + 5, column, "A2_Dataset")
            worksheet.write(row_excel + 6, column, "A2_Predicted")

        if row['predicted'] == 'A1' or row['predicted'] == 'R' or row['predicted'] == 'A2':
            new_word = True
            for s, p, o in g.triples((URIRef(row['word_id']), URIRef(base_uri+"word"), None)):
                #worksheet.write(row_excel - 6, column+1, o.__str__())
                word_id = row['word_id'].split("_")[-1]
                valid_paths = [[o.__str__(), word_id]]
                valid_paths = fetch_root_word_pair(g, base_uri, row['word_id'], valid_paths)
                sentences_paths = order_word_paths(valid_paths)
                for sentence in sentences_paths:
                    if row['predicted'] == 'A1':
                        if new_word:
                            worksheet.write(row_excel - 6, column + a1_k, sentence.strip(), format)
                            a1_k += 2
                            new_word = False
                        else:
                            worksheet.write(row_excel - 6, column + a1_k, sentence.strip())
                            a1_k += 2
                    elif row['predicted'] == 'R':
                        if new_word:
                            worksheet.write(row_excel - 4, column + r_k, sentence.strip(), format)
                            r_k += 2
                            new_word = False
                        else:
                            worksheet.write(row_excel - 4, column + r_k, sentence.strip())
                            r_k += 2
                    elif row['predicted'] == 'A2':
                        if new_word:
                            worksheet.write(row_excel - 2, column + a2_k, sentence.strip(), format)
                            a2_k += 2
                            new_word = False
                        else:
                            worksheet.write(row_excel - 2, column + a2_k, sentence.strip())
                            a2_k += 2

    workbook.close()


def check_if_punct(g, word_uri, base_uri):
    for s, p, o in g.triples((URIRef(word_uri), URIRef(base_uri + "pos"), None)):
        if o.__str__().split("#")[-1] == "PUNCT":
            return True
        else:
            return False


def convert_results_benchmark(base_uri, graph_name, connection_string, read_path, write_path, threshold):
    qb = QueryBuilder(base_uri, graph_name)
    results = pd.read_excel(read_path)
    workbook = xls.Workbook(write_path)
    worksheet = workbook.add_worksheet()
    row_excel = 0
    column = 0

    format = workbook.add_format()
    format.set_pattern(1)
    format.set_bg_color('gray')

    worksheet.write(row_excel, column, "Sentence")
    worksheet.write(row_excel+1, column, "A1_Predicted")
    worksheet.write(row_excel+2, column, "R_Predicted")
    worksheet.write(row_excel+3, column, "A2_Predicted")
    a1_k = 1
    a2_k = 1
    r_k = 1
    current_sentence = 0
    for index, row in results.iterrows():
        current_sent_id = row['word_id'].split("_")[-2]
        if current_sent_id != current_sentence:
            a1_k = 1
            a2_k = 1
            r_k = 1
            current_sentence = current_sent_id
            current_sentence_uri = base_uri + "Sentence_0_" + current_sentence
            #Fetch the sentence details
            triple_list = []
            g = fetch_graph(qb, current_sentence, connection_string)
            #Fetch sentence text
            worksheet.write(row_excel, column + 1, current_sentence)
            for s, p, o in g.triples((URIRef(current_sentence_uri), URIRef(base_uri+"senttext"), None)):
                graph = list_conll_subgraph(graph=g, root_node=s, transverse_by=URIRef(base_uri + "depGraph"),
                                            order_by=URIRef(base_uri + "id"),
                                            main_uri=base_uri)
                worksheet.write(row_excel, column+2, o.__str__())
                worksheet.write(row_excel, column+4, graph.__str__())

            column = 0
            row_excel += 5
            worksheet.write(row_excel, column, "Sentence")
            worksheet.write(row_excel + 1, column, "A1_Predicted")
            worksheet.write(row_excel + 2, column, "R_Predicted")
            worksheet.write(row_excel + 3, column, "A2_Predicted")

        if row['predicted'] == 'A1' or row['predicted'] == 'R' or row['predicted'] == 'A2':
            new_word = True
            #print("Type: ", row['predicted'])
            for s, p, o in g.triples((URIRef(row['word_id']), URIRef(base_uri+"word"), None)):
                #worksheet.write(row_excel - 6, column+1, o.__str__())
                #print(s, p, o)
                if not check_if_punct(g, row['word_id'], base_uri):
                    word_id = row['word_id'].split("_")[-1]
                    valid_paths = [[o.__str__(), word_id]]
                    valid_paths = fetch_root_word_pair(g, base_uri, row['word_id'], valid_paths)
                    #print(valid_paths)
                    sentences_paths = order_word_paths(valid_paths)
                    #print(sentences_paths)
                    #print("-------")
                    for sentence in sentences_paths:
                        if row['predicted'] == 'A1' and row['A1'] >= threshold:
                            if new_word:
                                worksheet.write(row_excel - 4, column + a1_k, o.__str__(), format)
                                worksheet.write(row_excel - 4, column + a1_k + 2, sentence.strip())
                                a1_k += 4
                                new_word = False
                            else:
                                worksheet.write(row_excel - 4, column + a1_k, sentence.strip())
                                a1_k += 2
                        elif row['predicted'] == 'R' and row['R'] >= threshold:
                            if new_word:
                                worksheet.write(row_excel - 3, column + r_k, o.__str__(), format)
                                worksheet.write(row_excel - 3, column + r_k+2, sentence.strip())
                                r_k += 4
                                new_word = False
                            else:
                                worksheet.write(row_excel - 3, column + r_k, sentence.strip())
                                r_k += 2
                        elif row['predicted'] == 'A2' and row['A2'] >= threshold:
                            if new_word:
                                worksheet.write(row_excel - 2, column + a2_k, o.__str__(), format)
                                worksheet.write(row_excel - 2, column + a2_k+2, sentence.strip())
                                a2_k += 4
                                new_word = False
                            else:
                                worksheet.write(row_excel - 2, column + a2_k, sentence.strip())
                                a2_k += 2

    workbook.close()


config_file = open('Benchmarks/Configs/GraphML_OpenIE.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)

convert_results_benchmark(base_uri=config_data['connection'][0]['base_uri'], graph_name=config_data['connection'][0]['graph_name'], connection_string=config_data['connection'][0]['connection_uri']
                          , read_path="Benchmarks/Results/CaRB.xlsx", write_path ="Benchmarks/Results/excel/Roots_Only/CaRB_Tests.xlsx", threshold=0)
