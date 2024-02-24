import torch
import pandas as pd
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
import yaml
from GraphBuilderLinkPred import get_graph
from Model import GAT, GNN, Spline, ModelLink
import random
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

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


def wandb_data(data):
    wandb.init(project='Competition_Task1')
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = data.num_features
    summary["data"]["num_classes"] = data.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    #summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)


config_file = open('configs/graphml_conf_task1_linkpred.yaml', 'r')
training_config = open('configs/training_conf.yaml', 'r')

config_data = yaml.load(config_file, Loader=yaml.FullLoader)
training_config = yaml.load(training_config, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']

if enable_wandb:
    import wandb

data, nodes, mapped_uris = get_graph([*range(1, 2000, 1)], config_data, test=False)
data['word'].node_id = torch.arange(len(mapped_uris['word']))
data['word'].num_nodes = len(mapped_uris['word'])
data['sentence'].node_id = torch.arange(len(mapped_uris['sentence']))
data['sentence'].num_nodes = len(mapped_uris['sentence'])
data['Entity'].node_id = torch.arange(len(mapped_uris['Entity']))
data['Entity'].num_nodes = len(mapped_uris['Entity'])

print(data)
#data = T.ToUndirected()(data)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("word", "softwareMention", "Entity"),
)
train_data, val_data, test_data = transform(data)
edge_label_index = train_data["word", "softwareMention", "Entity"].edge_label_index
edge_label = train_data["word", "softwareMention", "Entity"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("word", "softwareMention", "Entity"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)

model = ModelLink(64,data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 50):
    total_loss = total_examples = 0
    for sampled_data in tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["word", "softwareMention", "Entity"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

model.eval()
edge_label_index = val_data["word", "softwareMention", "Entity"].edge_label_index
edge_label = val_data["word", "softwareMention", "Entity"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("word", "softwareMention", "Entity"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

sampled_data = next(iter(val_loader))
preds = []
ground_truths = []
for sampled_data in tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        out = model(sampled_data)
        preds.append(out)
        ground_truths.append(sampled_data["word", "softwareMention", "Entity"].edge_label)

pred = torch.cat(preds, dim=0).cpu().numpy()
print(pred)
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
print(ground_truth)
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")
score = f1_score(ground_truth, pred)
print(score)
accuracy = accuracy_score(ground_truth, pred)
print(accuracy)

#data, targets = get_graph(random.sample(range(30000), 2000), config_data)
# ----------------- LOAD AND SAVE DATA WHEN NEEDED -------------------------
#torch.save(data, training_config['data_file'])
#targets.to_pickle(training_config['targets_file'])
# data = torch.load(training_config['data_file'])
# targets = pd.read_pickle(training_config['targets_file'])

# model = GNN(hidden_channels=64, out_channels=data.num_classes)
# model = to_hetero(model, data.metadata(), aggr='sum')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)
#
# if enable_wandb:
#     wandb_data(data)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Device: '{device}'")
# model = model.to(device)
# data = data.to(device)
#
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
#     pbar.set_description(f"Epoch {i} with Loss: {loss_final} -- Best Loss: {best_loss} on Epoch {best_epoch}", refresh=True)
#     #print("Loss: ", loss_final)
#
# # data_test, targets_test = get_graph([*range(10000, 10500, 1)], config_data, test = True, targets_test = targets)
# # # ----------------- LOAD AND SAVE DATA WHEN NEEDED -------------------------
# # torch.save(data, training_config['test_data_file'])
# # targets.to_pickle(training_config['test_targets_file'])
# # data_split = torch.load(training_config['data_file'])
# # targets = pd.read_pickle(training_config['targets_file'])
#
# data_test = data_test.to(device)
# test_acc, ground_truth, predictions, predict_percents = test(data_test = data_test)
# ground_truth = ground_truth.cpu().tolist()
# predictions = predictions.cpu().tolist()
# predict_percents = predict_percents.cpu().tolist()
#
# if enable_wandb:
#     wandb.summary["gat/accuracy"] = test_acc
#     wandb.log({"gat/accuracy": test_acc})
#     cm = wandb.plot.confusion_matrix(
#         y_true=ground_truth, preds=predictions, class_names=targets['originalId']
#     )
#     wandb.log({"gat/conf_mat": cm})
#     wandb.log({"gat/roc": wandb.plot.roc_curve(ground_truth, predict_percents, labels=targets_test['originalId'])})
#     wandb.log({"gat/pr": wandb.plot.pr_curve(ground_truth, predict_percents, labels=targets_test['originalId'])})