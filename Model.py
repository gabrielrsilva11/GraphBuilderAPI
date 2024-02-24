from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear, SAGEConv, SplineConv, to_hetero
import torch
import torch.nn.functional as F
from torch_geometric.nn.norm.batch_norm import BatchNorm


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
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = SAGEConv((-1, -1), int(hidden_channels/2))
        self.batch_norm2 = BatchNorm(int(hidden_channels/2))
        self.lin2 = Linear(-1, int(hidden_channels/2))
        self.conv3 = SAGEConv((-1, -1), int(hidden_channels/4))
        self.batch_norm3 = BatchNorm(int(hidden_channels/4))
        self.lin3 = Linear(-1, int(hidden_channels/4))
        self.conv4 = SAGEConv((-1, -1), out_channels)
        self.lin4 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)+self.lin1(x)
        x = x.relu()
        #x = F.dropout(x, p=0.5)
        x = self.batch_norm1(x)

        x = self.conv2(x, edge_index) + self.lin2(x)
        x = x.relu()
        x = self.batch_norm2(x)

        x = self.conv3(x, edge_index) + self.lin3(x)
        x = x.relu()
        x = self.batch_norm3(x)

        x = self.conv4(x, edge_index) + self.lin4(x)
        return x

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

class Spline(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = SplineConv((-1,-1), 16, dim=1, kernel_size=5)
        self.conv2 = SplineConv(16, 32, dim=1, kernel_size=5)
        self.conv3 = SplineConv(32, 64, dim=1, kernel_size=7)
        self.conv4 = SplineConv(64, 128, dim=1, kernel_size=7)
        self.conv5 = SplineConv(128, 128, dim=1, kernel_size=11)
        self.conv6 = SplineConv(128, out_channels, dim=1, kernel_size=11)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.elu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        x = F.elu(self.conv5(x, edge_index))
        x = self.conv6(x, edge_index)
        x = F.dropout(x)
        return F.log_softmax(x, dim=1)

class GNN_link(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_word: Tensor, x_entity: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_word = x_word[edge_label_index[0]]
        edge_feat_entity = x_entity[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_word * edge_feat_entity).sum(dim=-1)

class ModelLink(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.word_lin = torch.nn.Linear(23, hidden_channels)
        self.word_emb = torch.nn.Embedding(data["word"].num_nodes, hidden_channels)
        self.entity_emb = torch.nn.Embedding(data["Entity"].num_nodes, hidden_channels)
        self.sentence_emb = torch.nn.Embedding(data["sentence"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN_link(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "word": self.word_lin(data['word'].x) + self.word_emb(data["word"].node_id),
            "Entity": self.entity_emb(data["Entity"].node_id),
            "sentence": self.entity_emb(data["sentence"].node_id)
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["word"],
            x_dict["Entity"],
            # x_dict["sentence"],
            data["word", "softwareMention", "Entity"].edge_label_index,
            # data["word", "depGraph", "word"].edge_label_index,
            # data["word", "previousWord", "word"].edge_label_index,
            # data["word", "nextWord", "word"].edge_label_index,
            # data["word", "fromSentence", "sentence"].edge_label_index,
            # data["sentence", "nextSentence", "sentence"].edge_label_index,
            # data["sentence", "previousSentence", "sentence"].edge_label_index,
        )
        return pred
