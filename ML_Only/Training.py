import torch
import pandas as pd
from torch_geometric.nn import to_hetero
from tqdm import tqdm
import yaml
from FetchData import get_graph
from Model import GAT, GNN, Spline, Net
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import xlsxwriter as xls
import torch.nn.functional as F
import itertools
import pickle


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(input=out['word'], target=data['word'].y)#, weight=weights)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(model, data_test):
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


def wandb_data(data, name, training_config):
    run = wandb.init(project= training_config['project_name'], name=name)
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = data.num_features
    summary["data"]["num_classes"] = data.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    summary["data"]["file_name"] =  training_config['model_file']
    summary["data"]["training_data"] =  training_config['data_file']
    summary["data"]["test_data"] =  training_config['test_data_file']
    #summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)
    return run


def fetch_data(ids_to_fetch, training_config, fetch_type, targets_test = False):
    if fetch_type == "Train" and training_config['load_data_train']:
        data = torch.load(training_config['data_file'])
        targets = pd.read_pickle(training_config['targets_file'])
        return data, targets, False, False
    elif fetch_type == "Test" and training_config['load_data_test']:
        data = torch.load(training_config['test_data_file'])
        targets = pd.read_pickle(training_config['test_targets_file'])
        with open(training_config['test_indexes_file'], "rb") as f:
            word_index = pickle.load(f)
        return data, targets, False, word_index
    else:
        if fetch_type == "Train":
            data, targets, mapped_uris, word_index = get_graph(ids_to_fetch, config_data, test='Train',
                                                           embedding=training_config['embeddings'])
        elif fetch_type == "Test":
            data, targets, mapped_uris, word_index = get_graph(ids_to_fetch, config_data, test = 'Test',
                                                               targets_test = targets_test, embedding=training_config['embeddings'])

    return data, targets, mapped_uris, word_index


def write_to_excel(file_name, word_indexes, targets, results_percentage):
    workbook = xls.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, "word_id")
    worksheet.write(0, 1, "predicted")
    worksheet.write(0, 2, "real")
    worksheet.write(0, 3, "No")
    worksheet.write(0, 4, "R")
    worksheet.write(0, 5, "A1")
    worksheet.write(0, 6, "A2")

    row = 0
    for i in range(1, len(word_indexes)):
        column = 0
        worksheet.write(row, column, word_indexes[i])
        worksheet.write(row, column+1, targets['originalId'][predictions[i]])
        worksheet.write(row, column+2, targets['originalId'][ground_truth[i]])
        for j in range(0, len(results_percentage[i])):
            worksheet.write(row, column+3+j, results_percentage[i][j])
        row += 1

    workbook.close()


config_file = open('Configs/GraphML_OpenIE.yaml', 'r')
training_config = open('Configs/training_conf.yaml', 'r')

config_data = yaml.load(config_file, Loader=yaml.FullLoader)
training_config = yaml.load(training_config, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']

if enable_wandb:
    import wandb

#Fetch Training Data
training_ids_to_fetch = [*range(0, 10000, 1)]
data, targets, mapped_uris_train, word_indexes_train = fetch_data(training_ids_to_fetch, training_config, fetch_type = 'Train')

#Fetch Testing data
testing_ids_to_fetch = [*range(3000, 5000, 1)]
data_test, targets_test, mapped_uris, word_indexes = fetch_data(testing_ids_to_fetch, training_config, fetch_type='Test', targets_test = targets)

if training_config['save_data_train']:
    torch.save(data, training_config['data_file'])
    targets.to_pickle(training_config['targets_file'])

if training_config['save_data_test']:
    torch.save(data, training_config['test_data_file'])
    targets.to_pickle(training_config['test_targets_file'])
    with open(training_config['test_indexes_file'], 'wb') as f:
        pickle.dump(word_indexes, f)

#Send data to appropriate device (Usually GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

data = data.to(device)
# weights = torch.FloatTensor([0.05545, 14.1463])
# weights = weights.to(device)
data_test = data_test.to(device)

# print(targets)
# print(targets_test)
print("Training data: ")
print(targets)
print(data)

print("Testing data: ")
print(targets_test)
print(data_test)

current_best_recall = -1
current_best_metrics = []
current_best_params = []

#
learning_rate = [0.001]
hidden_channels = [32]
aggr = ['sum']
parameter_list = [learning_rate, hidden_channels, aggr]
parameters_combination = list(itertools.product(*parameter_list))
k = -1

if not training_config['train_new']:
    model = torch.load(training_config['model_file'])
    model = model.to(device)

for params in parameters_combination:
    k += 1
    print(params, k)

    if enable_wandb:
        name = training_config['run_name'] + str(k)
        run = wandb_data(data, name, training_config)

    if training_config['train_model']:
        model = GNN(hidden_channels=params[1], out_channels=data.num_classes)
        model = to_hetero(model, data.metadata(), aggr=params[2])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params[0])

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        epochs = training_config['epochs']
        pbar = tqdm(range(epochs), desc="Training Model")
        best_loss = 9999999
        best_epoch = 0
        for i in pbar:
            loss_final = train()
            if enable_wandb:
                wandb.log({"gat/loss": loss_final})

            if best_loss > loss_final:
                best_loss = loss_final
                best_epoch = i
        print("Loss: ", best_loss, "Epoch: ", best_epoch)

    # Testing Portion of the script
    test_acc, ground_truth, predictions, predict_percents = test(model, data_test = data_test)

    results_percentage = F.softmax(predict_percents).cpu().tolist()
    ground_truth = ground_truth.cpu().tolist()
    predictions = predictions.cpu().tolist()
    predict_percents = predict_percents.cpu().tolist()
    precision, recall, f1, support = precision_recall_fscore_support(ground_truth, predictions, average='macro')
    print("Precision: ", precision, " Recall: ", recall, " F1: ", f1)

    if recall > current_best_recall:
        torch.save(model, training_config['model_file'])
        current_best_params = params
        current_best_recall = recall
        current_best_metrics = [k, f1, precision, recall]
    cm = confusion_matrix(ground_truth, predictions)

    # Wandb logging
    if enable_wandb:
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

    #Write Results to an Excel file
    file_name = 'Results/Resultados_GridSearch'+str(k)+'.xlsx'
    write_to_excel(file_name, word_indexes, targets, results_percentage)

    run.finish()

print(current_best_params)
print(current_best_recall)
print(current_best_metrics)