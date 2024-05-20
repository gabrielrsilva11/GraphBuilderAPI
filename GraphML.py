import torch
import pandas as pd
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from GraphBuildWithConfig import get_graph
from Model import GAT, GNN, Spline, Net
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def train_batch():
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['word'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out['word'][:batch_size],
                               batch['word'].y[:batch_size])#, weight=weights)
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(input=out['word'], target=data['word'].y)#, weight=weights)
    loss.backward()
    optimizer.step()
    return float(loss)

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
    wandb.init(project='Slate2024_Embeddings', name=name)
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = data.num_features
    summary["data"]["num_classes"] = data.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    #summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)


config_file = open('configs/graphml_conf_task1_coreference.yaml', 'r')
training_config = open('configs/training_conf.yaml', 'r')

config_data = yaml.load(config_file, Loader=yaml.FullLoader)
training_config = yaml.load(training_config, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']

if enable_wandb:
    import wandb

# data, targets = get_graph([*range(6482, 9482, 1)], config_data, test=False, embedding=False)
#data, targets = get_graph(random.sample(range(30000), 2000), config_data)
# ----------------- LOAD AND SAVE DATA WHEN NEEDED -------------------------
# torch.save(data, training_config['data_file'])
# targets.to_pickle(training_config['targets_file'])
data = torch.load(training_config['data_file'])
targets = pd.read_pickle(training_config['targets_file'])

model = GNN(hidden_channels=64, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='max')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    #out = model(data)

if enable_wandb:
    name = "ConllEmbeddings_3kTrain_1kTest_5classes_Embeddings"
    wandb_data(data, name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
data = data.to(device)

# epochs = training_config['epochs']
# pbar = tqdm(range(epochs), desc="Training Model")
# best_loss = 9999999
# best_epoch = 0
# for i in pbar:
#     loss_final = train()
#     if enable_wandb:
#         wandb.log({"gat/loss": loss_final})
#     if best_loss > loss_final:
#         best_loss = loss_final
#         best_epoch = i
#     #if i%50 == 0:
#     pbar.set_description(f"Epoch {i} Loss: {loss_final} -- Best: {best_loss} Epoch {best_epoch}", refresh=True)
#     #print("Loss: ", loss_final)
#
# torch.save(model, training_config['model_file'])
model = torch.load(training_config['model_file'])
print(model.parameters)
model.eval()
# data_test, targets_test = get_graph([*range(10000, 11000, 1)], config_data, test = True, targets_test = targets, embedding=True)
# # # ----------------- LOAD AND SAVE DATA WHEN NEEDED -------------------------
# torch.save(data_test, training_config['test_data_file'])
# targets_test.to_pickle(training_config['test_targets_file'])
data_test = torch.load(training_config['test_data_file'])
targets_test = pd.read_pickle(training_config['test_targets_file'])

data_test = data_test.to(device)
test_acc, ground_truth, predictions, predict_percents = test(data_test = data_test)
ground_truth = ground_truth.cpu().tolist()
predictions = predictions.cpu().tolist()
predict_percents = predict_percents.cpu().tolist()

precision, recall, f1, support = precision_recall_fscore_support(ground_truth, predictions, average='macro')
print("Precision: ", precision, " Recall: ", recall, " F1: ", f1)
cm = confusion_matrix(ground_truth, predictions)
print(cm)

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
