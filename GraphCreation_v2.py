import string
from InsertData_PreProcess import CreateGraph
import yaml
import copy
from nltk.corpus import stopwords

#Function to pre-process a sentence. The output should be:
# The full sentence and a list of any extra info to be added.
# This list should be blank if there are no extra annotations and you want the default graph.
def preprocess_sentence(sentence):
    #text_nohtml = re.sub(r'http\S+', '', sentence)
    #sentences_split = sentence.replace("\n", "").split(" ")
    sentences_split = sentence.split(" ")
    final_list = []
    for sentence_annotation in sentences_split:
        sentence_annotation = sentence_annotation.split("|")
        if sentence_annotation[-1] == "O":
            sentence_annotation[-1] = 'No'
            sentence_annotation.append("No")
        else:
            sentence_entity = sentence_annotation[-1].split("-")
            sentence_annotation[-1] = sentence_entity[-1]
            sentence_annotation.append("Yes")
        # if "/" in sentence_annotation[0]:
        #     split_words = sentence_annotation[0].split("/")
        #     for word in split_words:
        #         copy_annotations = copy.deepcopy(sentence_annotation)
        #         copy_annotations[0] = word
        #         final_list.append(copy_annotations)
        # else:
        final_list.append(sentence_annotation)
    return final_list


# def preprocess_sentence_datasets(sentences):
#     final_list = []
#     for sentence in sentences:
#         for i in range(0, len(sentence['tokens'])):
#             final_list.append([sentence['tokens'][i], sentence['pos_tags'][i], sentence['chunk_tags'][i], sentence['ner_tags'][i]])
#     return final_list


config_file = open("configs/create_graph.yaml", 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)

graph = CreateGraph(folder=config_data['folder'], graph_name=config_data['graph_name'], extra_connetions=config_data['extra_connections']['connections'],
                    connection_string=config_data['connection'], main_uri=config_data['uri'], language=config_data['language'],
                    preprocessing=preprocess_sentence, in_memory=config_data['in_memory'])

graph.create_graph(save_file=config_data['save_file'], coref=False)
#graph.create_graph_from_datasets(dataset_name = "conll2003", save_file=config_data['save_file'], coref=False)
