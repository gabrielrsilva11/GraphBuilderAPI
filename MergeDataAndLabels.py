# import csv
# import re
#
# path_data_file = "Data_to_process/OpenIE/s2_pt/train.tsv"
# merged_file = open("Data_to_process/OpenIE/s2_pt/Merged/train.txt", "w")
# error_log = open("Data_to_process/OpenIE/s2_pt/error.txt", "w")
# regex_1 = r"\>(.*)"
# regex_2 = r"(?<=>)(.*?)(?=<)"
# sentence_list = []
# with open(path_data_file) as fd:
#     rd = csv.reader(fd, delimiter="\t", quotechar='"')
#     for row in rd:
#         matches = re.findall(regex_1, row[0])
#         matches_2 = re.findall(regex_2, row[1])
#         sentence = matches[0]
#         try:
#             sentence = str.replace(sentence, matches_2[0], " <a1>"+matches_2[0]+"<\\a1> ")
#             sentence = str.replace(sentence, matches_2[2], " <r>" + matches_2[2] + "<\\r> ")
#             sentence = str.replace(sentence, matches_2[4], " <a2>" + matches_2[4] + "<\\a2> ")
#             sentence_list.append(sentence)
#         except:
#             error_log.write(matches[0] + "\n" + row[1] + "\n\n")
#         # merged_file.write("{}|{} ".format(matches[0], matches[1]))
#         # merged_file.write("\n")
#
# for sentence in sentence_list:
#     notation = "O"
#     sentence = sentence.strip()
#     for word in sentence.split(" "):
#         if word == "<a1>":
#             notation = "A1"
#         elif word == "<a2>":
#             notation = "A2"
#         elif word == "<r>":
#             notation = "R"
#         elif word == "<\\r>" or word == "<\\a1>" or word == "<\\a2>":
#             notation = "O"
#         else:
#             merged_file.write("{}|{} ".format(word, notation))
#     merged_file.write("\n")

# with open(path_data_file) as data_file, open(path_annotations_file) as anno_file:
#     for data, label in zip(data_file, anno_file):
#         data_split = data.replace("\n", "").split(" ")
#         label_split = label.replace("\n", "").split(" ")
#         for i in range(0, len(data_split)):
#             merged_file.write("{}|{} ".format(data_split[i], label_split[i]))
#         merged_file.write("\n")

# def find_triples(sentence, triple_part, type):
#     triple_part_split = triple_part.split(" ")
#     triple_part = triple_part.replace("\n","")
#     new_triple_part = ""
#     for word in triple_part_split:
#         word = word.strip()
#         new_triple_part = new_triple_part+word+"|"+type+" "
#     sentence = sentence.replace(triple_part, new_triple_part)
#     return sentence.replace("  ", " ")
#
#
# file_path = "/home/grsilva/GraphBuilderAPI_v2/Data_to_process/OpenIE/CaRB/gold_dev.tsv"
# new_file = "/home/grsilva/GraphBuilderAPI_v2/Data_to_process/OpenIE/CaRB/Processed/gold_dev.txt"
# f = open(file_path, "r")
# i = 0
# with open(new_file, "w") as f_write:
#     for line in f:
#         split_sent = line.split("\t")
#         sentence = split_sent[0]
#         subject = split_sent[2]
#         predicate = split_sent[1]
#         object = split_sent[3]
#         sentence = find_triples(sentence, subject, "A1")
#         sentence = find_triples(sentence, predicate, "R")
#         sentence = find_triples(sentence, object, "A2")
#         f_write.write(sentence+"\n")

file_path = "/home/grsilva/GraphBuilderApi/Data_to_process/SOMD2025/train_texts.txt"
relations_path = "/home/grsilva/GraphBuilderApi/Data_to_process/SOMD2025/train_entities.txt"
merged_file = "/home/grsilva/GraphBuilderApi/Data_to_process/SOMD2025/Merged_Train.txt"
teste_file_path = "/home/grsilva/GraphBuilderApi/Data_to_process/SOMD2025/test_texts.txt"
merged_test_file = "/home/grsilva/GraphBuilderApi/Data_to_process/SOMD2025/Merged_Test.txt"
# Read the contents of train_texts.txt
with open(file_path, 'r') as texts_file:
    texts = texts_file.readlines()

# Read the contents of train_relations.txt
with open(relations_path, 'r') as relations_file:
    relations = relations_file.readlines()

# Ensure both files have the same number of lines
if len(texts) != len(relations):
    raise ValueError("The number of lines in train_texts and train_relations must be the same.")

# Write the merged data to a new file
# with open(merged_file, 'w') as output_file:
#     for i in range(0, len(relations)):
#         texts_split = texts[i].split(" ")
#         relations_split = relations[i].split(" ")
#         str_builder = ""
#         for merge_i in range(0, len(texts_split)):
#             if merge_i == len(texts_split)-1:
#                 str_builder += texts_split[merge_i].strip()+"|"+relations_split[merge_i]
#             else:
#                 str_builder += texts_split[merge_i].strip()+"|"+relations_split[merge_i]+" "
#         output_file.write(str_builder)

with open(teste_file_path, 'r') as texts_file:
    teste_texts = texts_file.readlines()

with open(merged_test_file, 'w') as output_file:
    for sentence in teste_texts:
        sentence = sentence.strip("\n")
        words = sentence.split(" ")
        str_builder = ""
        for word in words:
            str_builder += word+"|O "
        output_file.write(str_builder+"\n")



print("Files have been successfully merged into merged_output.txt")
