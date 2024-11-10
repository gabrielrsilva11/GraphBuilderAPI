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

config_file = open('configs/OpenIE/GraphML_OpenIE.yaml', 'r')
training_config = open('configs/training_conf.yaml', 'r')

config_data = yaml.load(config_file, Loader=yaml.FullLoader)
training_config = yaml.load(training_config, Loader=yaml.FullLoader)

def test(data_test):
    model.eval()
    out = model(data_test.x_dict, data_test.edge_index_dict)
    predictions = out['word'].argmax(dim=1)
    truth = data_test['word'].y
    test_correct = predictions == truth  # Check against ground-truth labels.
    test_correct = test_correct.tolist()
    test_acc = int(sum(test_correct)) / len(test_correct)#int(data_test['word'].sum())  # Derive ratio of correct predictions.
    #if enable_wandb:
    #    embedding_to_wandb(out['word'][mask], unique_targets, color=truth, key="gat/summary")
    return test_acc, truth, predictions, out['word']

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
    summary["data"]["num_classes"] = data.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    #summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)
    return run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Load training targets
targets = pd.read_pickle(training_config['targets_file'])
#Load testing targets
data_test, targets_test, mapped_uris, word_indexes = get_graph([*range(0, 1500, 1)], config_data, test = True, targets_test = targets, embedding=training_config['embeddings'])
data_test = data_test.to(device)
#Load model
model = torch.load(training_config['model_file'])
#Apply weights
weights = torch.FloatTensor([0.05545, 14.1463, 14.1463, 14.1463])
weights = weights.to(device)
#Initialize run
run = wandb_data(data_test, "First_CaRB_Test")

test_acc, ground_truth, predictions, predict_percents = test(data_test = data_test)
results_percentage = F.softmax(predict_percents).cpu().tolist()
ground_truth = ground_truth.cpu().tolist()
predictions = predictions.cpu().tolist()
predict_percents = predict_percents.cpu().tolist()
precision, recall, f1, support = precision_recall_fscore_support(ground_truth, predictions, average='macro')
cm = confusion_matrix(ground_truth, predictions)
print("Precision: ", precision, " Recall: ", recall, " F1: ", f1)
print(cm)

#Finalize run logging
wandb.summary["gat/accuracy"] = test_acc
wandb.log({"gat/accuracy": test_acc,
           "gat/precision": precision,
           "gat/recall": recall,
           "gat/f1": f1})
cm = wandb.plot.confusion_matrix(
    y_true=ground_truth, preds=predictions, class_names=targets['originalId']
)
wandb.log({"gat/conf_mat": cm})
wandb.log({"gat/roc": wandb.plot.roc_curve(ground_truth, predict_percents, labels=targets_test['originalId'])})
wandb.log({"gat/pr": wandb.plot.pr_curve(ground_truth, predict_percents, labels=targets_test['originalId'])})
run.finish()

workbook = xls.Workbook('Results/Validation/GridSearch/Resultados_GridSearch_PT_Teste.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "word_id")
worksheet.write(0, 1, "predicted")
worksheet.write(0, 2, "real")
worksheet.write(0, 3, "No")
worksheet.write(0, 4, "R")
worksheet.write(0, 5, "A1")
worksheet.write(0, 6, "A2")

row = 0
for i in range(0, len(word_indexes)):
    column = 0
    #fetch the word attributes
    #if ground_truth[i] != predictions[i]:
        #print(word_indexes[i], targets['originalId'][predictions[i]], targets['originalId'][ground_truth[i]])
    worksheet.write(row, column, word_indexes[i])
    worksheet.write(row, column+1, targets_test['originalId'][predictions[i]])
    worksheet.write(row, column+2, targets_test['originalId'][ground_truth[i]])
    for j in range(0, len(results_percentage[i])):
        worksheet.write(row, column+3+j, results_percentage[i][j])
    row += 1
workbook.close()
