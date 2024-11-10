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

def find_triples(sentence, triple_part, type):
    triple_part_split = triple_part.split(" ")
    triple_part = triple_part.replace("\n","")
    new_triple_part = ""
    for word in triple_part_split:
        word = word.strip()
        new_triple_part = new_triple_part+word+"|"+type+" "
    sentence = sentence.replace(triple_part, new_triple_part)
    return sentence.replace("  ", " ")


file_path = "Data_to_process/OpenIE/CaRB/gold_test.tsv"
new_file = "Data_to_process/OpenIE/CaRB/gold_test_processed_V2.txt"
f = open(file_path, "r")
i = 0
with open(new_file, "w") as f_write:
    for line in f:
        split_sent = line.split("\t")
        sentence = split_sent[0]
        subject = split_sent[2]
        predicate = split_sent[1]
        object = split_sent[3]
        sentence = find_triples(sentence, subject, "S")
        sentence = find_triples(sentence, predicate, "P")
        sentence = find_triples(sentence, object, "O")
        f_write.write(sentence+"\n")
