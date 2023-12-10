from GraphConverterNetworkX import BuildNetworkx
from rdflib.namespace import RDF
from rdflib import URIRef, Literal, Graph
import torch
from sklearn import preprocessing
from torch_geometric.data import HeteroData
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero, SAGEConv
import torch.nn.functional as F
import tqdm
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
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


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

def train_batch():
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['word'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out['word'][:batch_size],
                               batch['word'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

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
    if enable_wandb:
        embedding_to_wandb(out['word'][mask], color=truth, key="gat/summary")
    return test_acc


def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    print(df)
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})

def wandb_data(dataset, data):
    wandb.init(project='WikinerPytorchGeo-attempt2')
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
    wandb.log(summary)

config_file = open('conf.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']

if enable_wandb:
    import wandb

#data = get_graph([*range(1, 3000, 1)], config_data)
data = torch.load("data/GraphData/1_to_3000.pt")
#print(data)
# CREATING THE TRAIN, TEST AND VAL MASKS
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
data_split = split(data)
print(data_split)

train_loader = NeighborLoader(
    data_split,
    num_neighbors=[15] * 2,
    batch_size=128,
    input_nodes=('word', data_split['word'].train_mask),
)

#Creating the model
model = GNN(hidden_channels=64, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

if enable_wandb:
    wandb_data(data, data_split)
    wandb.watch(model)

# Pass data onto GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
data_split = data_split.to(device)

for epoch in range(1, 100):
    print("---- Epoch ", epoch, "----")
    loss_final = train_batch()
    if enable_wandb:
        wandb.log({"gat/loss": loss_final})
    print("Loss: ", loss_final)

test_acc = test()
if enable_wandb:
    wandb.summary["gat/test_accuracy"] = test_acc
    wandb.log({"gat/accuracy": test_acc})

torch.save(model, "models/gnn.pt")
torch.save(data, "data/GraphData/1_to_3000.pt")

