from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
import torch
from sklearn import preprocessing
from torch_geometric.data import HeteroData
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
import tqdm
import yaml
from GraphBuildWithConfig import get_graph

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data_split.x_dict, data_split.edge_index_dict)
    mask = data_split['word'].train_mask
    loss = F.cross_entropy(out['word'][mask], data_split['word'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def test():
    model.eval()
    out = model(data_split.x_dict, data_split.edge_index_dict)
    mask = data_split['word'].test_mask
    predictions = out['word'][mask].argmax(dim=1)
    truth = data_split['word'].y[mask]
    test_correct = predictions == truth  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data_split['word'].test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

def wandb_data(dataset, data):
    wandb.init(project='WikinerPytorchGeo-attempt1')
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = dataset.num_features
    summary["data"]["num_classes"] = dataset.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    summary["data"]["has_isolated_nodes"] = data.has_isolated_nodes()
    summary["data"]["has_self_nodes"] = data.has_self_loops()
    summary["data"]["is_undirected"] = data.is_undirected()
    summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.summary = summary

config_file = open('conf.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']
if enable_wandb:
    import wandb

data = get_graph([*range(1, 100, 1)], config_data)
print(data)

# CREATING THE TRAIN, TEST AND VAL MASKS
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
data_split = split(data)
print(data_split)

wandb_data(data, data_split)

#Creating the model
model = GAT(hidden_channels=16, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Pass data onto GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
data_split = data_split.to(device)

for epoch in range(1, 200):
    print("---- Epoch ", epoch, "----")
    loss_final = train()
    if enable_wandb:
        wandb.log({"gat/loss": loss_final})
    print(loss_final)
    print("Loss: ", loss_final)

test_acc = test()
if enable_wandb:
    wandb.summary["gat/accuracy"] = test_acc
    wandb.log({"gat/accuracy": test_acc})

# out = model(data_split.x_dict, data_split.edge_index_dict)
# mask = data_split['word'].test_mask
#
# predictions = out['word'][data_split["word"].test_mask].argmax(dim=1)
# truth = data_split['word'].y[data_split['word'].test_mask]
# print(predictions)
# print(truth)

