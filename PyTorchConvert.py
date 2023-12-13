import torch
from torch_geometric.data import HeteroData
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero, SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import yaml
from GraphBuildWithConfig import get_graph


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, dropout=0.5)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, dropout=0.5)
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

def test(unique_targets):
    model.eval()
    out = model(data_split.x_dict, data_split.edge_index_dict)
    mask = data_split['word'].test_mask
    predictions = out['word'][mask].argmax(dim=1)
    truth = data_split['word'].y[mask]
    test_correct = predictions == truth  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data_split['word'].test_mask.sum())  # Derive ratio of correct predictions.
    print(test_acc)
    if enable_wandb:
        embedding_to_wandb(out['word'][mask], unique_targets, color=truth, key="gat/summary")
    return test_acc, truth, predictions, out['word'][mask]


def embedding_to_wandb(h, targets_index, color, key="embedding"):
    num_components = h.shape[-1]
    df = pd.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[targets_index['originalId'][i] for i in range(num_components)])
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
    summary["data"]["num_training_nodes"] = data['word'].train_mask.sum()
    wandb.log(summary)


config_file = open('conf.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)
enable_wandb = config_data['enable_wandb']

if enable_wandb:
    import wandb

data, targets = get_graph([*range(1, 3000, 1)], config_data)
# data = torch.load("data/GraphData/1_to_3000.pt")
# targets = pd.read_pickle("data/targets.pkl")

# model = torch.load("models/gnn.pt")
# print(data)
# CREATING THE TRAIN, TEST AND VAL MASKS
split = T.RandomNodeSplit(num_val=0.15, num_test=0.2)
data_split = split(data)
print(data_split)

train_loader = NeighborLoader(
    data_split,
    num_neighbors=[10] * 2,
    batch_size=256,
    input_nodes=('word', data_split['word'].train_mask),
)

#Creating the model
model = GAT(hidden_channels=64, out_channels=data.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)

if enable_wandb:
    wandb_data(data, data_split)
    #wandb.watch(model)

# Pass data onto GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
data_split = data_split.to(device)

epochs = 500

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
    pbar.set_description(f"Current Loss: {loss_final} -- Best Loss: {best_loss} on Epoch {best_epoch}", refresh=True)
    print("Loss: ", loss_final)

test_acc, ground_truth, predictions, predict_percents = test(unique_targets=targets)
ground_truth = ground_truth.cpu().tolist()
predictions = predictions.cpu().tolist()
predict_percents = predict_percents.cpu().tolist()

if enable_wandb:
    wandb.summary["gat/accuracy"] = test_acc
    wandb.log({"gat/accuracy": test_acc})
    cm = wandb.plot.confusion_matrix(
        y_true=ground_truth, preds=predictions, class_names=targets['originalId']
    )
    wandb.log({"gat/conf_mat": cm})
    wandb.log({"gat/roc": wandb.plot.roc_curve(ground_truth, predict_percents, labels=targets['originalId'])})
    wandb.log({"gat/pr": wandb.plot.pr_curve(ground_truth, predict_percents, labels=targets['originalId'])})

# torch.save(model, "models/gnn.pt")
torch.save(data, "data/GraphData/1_to_3000.pt")
targets.to_pickle("data/targets.pkl")

