import torch
import pandas as pd
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from GraphBuildWithConfig import get_graph, fetch_word_node_info
from Model import GAT, GNN, Spline, Net
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import xlsxwriter as xls
import torch.nn.functional as F
import itertools
import wandb
from ProcessResults import convert_results_benchmark

config_file = open('Configs/GraphML_OpenIE.yaml', 'r')
training_config = open('Configs/training_conf.yaml', 'r')

config_data = yaml.load(config_file, Loader=yaml.FullLoader)
training_config = yaml.load(training_config, Loader=yaml.FullLoader)

def embedding_to_wandb(h, targets_index, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[targets_index['originalId'][i] for i in range(num_components)])
    print(df)
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    print(df)
    wandb.log({key: df})

def wandb_data(data, name):
    run = wandb.init(project='Iberspeech_Runs_V3_PT', name=name)
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = data.num_features
    summary["data"]["num_classes"] = 4
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    #summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)
    return run

def test(data_test):
    model.eval()
    out = model(data_test.x_dict, data_test.edge_index_dict)
    predictions = out['word'].argmax(dim=1)
    return predictions, out['word']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_test_benchmark, mapped_uris, word_indexes = get_graph([*range(1, 613, 1)], config_data, test='Benchmark', embedding=training_config['embeddings'])
data_test_benchmark = data_test_benchmark.to(device)
print(data_test_benchmark)
targets = pd.read_pickle(training_config['targets_file'])
#Load model
model = torch.load(training_config['model_file'])
#Apply weights
weights = torch.FloatTensor([0.05545, 14.1463, 14.1463, 14.1463])
weights = weights.to(device)
model.eval()
#Initialize run
run = wandb_data(data_test_benchmark, "First_CaRB_Test")

predictions, predict_percents = test(data_test = data_test_benchmark)
predictions = predictions.cpu().tolist()
results_percentage = F.softmax(predict_percents).cpu().tolist()

workbook = xls.Workbook('Results/CaRB.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "word_id")
worksheet.write(0, 1, "predicted")
worksheet.write(0, 2, "No")
worksheet.write(0, 3, "R")
worksheet.write(0, 4, "A1")
worksheet.write(0, 5, "A2")
row = 1
for i in range(0, len(word_indexes)):
    column = 0
    worksheet.write(row, column, word_indexes[i])
    worksheet.write(row, column + 1, targets['originalId'][predictions[i]])
    for j in range(0, len(results_percentage[i])):
        worksheet.write(row, column + 2 + j, results_percentage[i][j])
    row += 1
workbook.close()
run.finish()

convert_results_benchmark(base_uri=config_data['connection'][0]['base_uri'], graph_name=config_data['connection'][0]['graph_name'], connection_string=config_data['connection'][0]['connection_uri']
                          , read_path="Results/CaRB.xlsx", write_path ="Results/excel/CaRB_Processed_v2.xlsx")