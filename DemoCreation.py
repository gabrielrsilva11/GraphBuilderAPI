from InsertData_Experiments_v2 import CreateGraph
import yaml
import copy


#Function to pre-process a sentence. The output should be:
# The full sentence and a list of any extra info to be added.
# This list should be blank if there are no extra annotations and you want the default graph.
def preprocess_sentence(sentence):
    sentences_split = sentence.replace("\n", "").split(" ")
    final_list = []
    for sentence_annotation in sentences_split:
        sentence_annotation = sentence_annotation.split("|")

        if sentence_annotation[-1] == "O":
            sentence_annotation[-1] = ''
            sentence_annotation.append("No")
        else:
            sentence_entity = sentence_annotation[-1].split("-")
            sentence_annotation[-1] = sentence_entity[-1]
            sentence_annotation.append("Yes")

        if "/" in sentence_annotation[0]:
            split_words = sentence_annotation[0].split("/")
            for word in split_words:
                copy_annotations = copy.deepcopy(sentence_annotation)
                copy_annotations[0] = word
                final_list.append(copy_annotations)
        else:
            final_list.append(sentence_annotation)
    return final_list


config_file = open("create_graph.yaml", 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)

graph = CreateGraph(folder="WikiNER_Original", graph_name=config_data['graph_name'], extra_connetions=config_data['extra_connections']['connections'],
                    connection_string=config_data['connection'], main_uri=config_data['uri'], language=config_data['language'], preprocessing=preprocess_sentence, in_memory=False)

graph.create_graph(save_file="WikiNERv5")