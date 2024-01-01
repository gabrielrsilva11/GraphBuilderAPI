from InsertData import CreateGraph
import yaml


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
        final_list.append(sentence_annotation)
    return final_list


config_file = open("create_graph.yaml", 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)

graph = CreateGraph(folder="WikiNER_Subsample", graph_name=config_data['graph_name'], extra_connetions=config_data['extra_connections']['connections'],
                    connection_string=config_data['connection'], main_uri=config_data['uri'], language=config_data['language'], preprocessing=preprocess_sentence)

graph.create_graph(in_memory=True, save_file="WikiNERv2")
