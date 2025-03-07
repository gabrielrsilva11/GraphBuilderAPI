from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear, SAGEConv, SplineConv, to_hetero, TopKPooling, aggr, HGTConv, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn.functional as F
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn import TransformerConv


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
    def __init__(self, hidden_channels, out_channels, aggregator):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr=aggregator)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = SAGEConv((-1, -1), int(hidden_channels/2), aggr=aggregator)
        self.batch_norm2 = BatchNorm(int(hidden_channels/2))
        self.lin2 = Linear(-1, int(hidden_channels/2))
        self.conv3 = SAGEConv((-1, -1), int(hidden_channels/4), aggr=aggregator)
        self.batch_norm3 = BatchNorm(int(hidden_channels/4))
        self.lin3 = Linear(-1, int(hidden_channels/4))
        self.conv4 = SAGEConv((-1, -1), out_channels, aggr=aggregator)
        self.lin4 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #+self.lin1(x)
        x = x.relu()
        #x = F.dropout(x, p=0.5)
        #x = self.batch_norm1(x)

        x = self.conv2(x, edge_index) #+ self.lin2(x)
        x = x.relu()
        #x = self.batch_norm2(x)

        x = self.conv3(x, edge_index) #+ self.lin3(x)
        x = x.relu()
        #x = self.batch_norm3(x)

        x = self.conv4(x, edge_index) + self.lin4(x)
        return x

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, meta, head_number):
        super().__init__()
        self.conv1 = HGTConv(in_channels = -1, out_channels = hidden_channels, metadata=meta, heads=head_number)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = HGTConv(-1, int(hidden_channels), metadata=meta, heads=head_number)
        self.batch_norm2 = BatchNorm(int(hidden_channels))
        self.lin2 = Linear(-1, int(hidden_channels))
        self.conv3 = HGTConv(-1, int(hidden_channels),metadata=meta, heads=head_number)
        self.batch_norm3 = BatchNorm(int(hidden_channels))
        self.lin3 = Linear(-1, int(hidden_channels))
        self.conv4 = HGTConv(-1, out_channels, metadata=meta, heads=1)
        self.lin4 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #+self.lin1(x)
        x = x.relu()
        #x = F.dropout(x, p=0.5)
        #x = self.batch_norm1(x)

        x = self.conv2(x, edge_index) #+ self.lin2(x)
        x = x.relu()
        #x = self.batch_norm2(x)

        x = self.conv3(x, edge_index) #+ self.lin3(x)
        x = x.relu()
        #x = self.batch_norm3(x)

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv((-1, -1), 512)
        self.pool1 = TopKPooling(512, ratio=0.8)
        self.conv2 = SAGEConv((-1, -1), 512)
        self.pool2 = TopKPooling(512, ratio=0.8)
        self.conv3 = SAGEConv((-1, -1), 512)
        self.pool3 = TopKPooling(512, ratio=0.8)
        #self.item_embedding = torch.nn.Embedding(num_embeddings=, embedding_dim=512)
        #self.lin0 = torch.nn.Linear(-1, 512)
        self.lin1 = Linear(-1, 512)
        self.lin2 = Linear(-1, 512 // 2)
        self.lin3 = Linear(-1, 7)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512//2)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.lin0(x.float())
        #x = self.item_embedding(x)
        #x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        z = self.pool1(x, edge_index, None, batch)
        x, edge_index, _, batch, _, _ = z
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x))
        x = self.lin3(x).squeeze(1)
        # print(x)

        return x

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, beta=True, heads=1):
        """
                Params:
                - input_dim: The dimension of input features for each node.
                - hidden_dim: The size of the hidden layers.
                - output_dim: The dimension of the output features (often equal to the
                    number of classes in a classification task).
                - num_layers: The number of layer blocks in the model.
                - dropout: The dropout rate for regularization. It is used to prevent
                    overfitting, helping the learning process remains generalized.
                - beta: A boolean parameter indicating whether to use a gated residual
                    connection (based on equations 5 and 6 from the UniMP paper). The
                    gated residual connection (controlled by the beta parameter) helps
                    preventing overfitting by allowing the model to balance between new
                    and existing node features across layers.
                - heads: The number of heads in the multi-head attention mechanism.
                """
        super(Transformer, self).__init__()
        # The list of transormer conv layers for the each layer block.
        self.num_layers = num_layers
        print(input_dim, hidden_dim, output_dim)
        conv_layers = [TransformerConv(input_dim, hidden_dim // heads, heads=heads, beta=beta)]
        conv_layers += [TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, beta=beta) for _ in
                        range(num_layers - 2)]
        # In the last layer, we will employ averaging for multi-head output by
        # setting concat to True.
        conv_layers.append(TransformerConv(hidden_dim, output_dim, heads=heads, beta=beta, concat=True))
        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

        # Probability of an element getting zeroed.
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets the parameters of the convolutional and normalization layers,
        ensuring they are re-initialized when needed.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        """
        The input features are passed sequentially through the transformer
        convolutional layers. After each convolutional layer (except the last),
        the following operations are applied:
        - Layer normalization (`LayerNorm`).
        - ReLU activation function.
        - Dropout for regularization.
        The final layer is processed without layer normalization and ReLU
        to average the multi-head results for the expected output.

        Params:
        - x: node features x
        - edge_index: edge indices.

        """
        for i in range(self.num_layers - 1):
          # Construct the network as shown in the model architecture.
          x = self.convs[i](x, edge_index)
          x = self.norms[i](x)
          x = F.relu(x)
          # By setting training to self.training, we will only apply dropout
          # during model training.
          x = F.dropout(x, p = self.dropout, training = self.training)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index)

        return x

class GNN_v2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, aggregator, num_layers, dropout):
        super(GNN_v2, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        conv_layers = [SAGEConv((-1, -1), hidden_channels, aggr=aggregator)]
        conv_layers += [SAGEConv((-1, -1), hidden_channels, aggr=aggregator) for _ in
                        range(num_layers - 2)]
        conv_layers.append(SAGEConv((-1, -1), out_channels, aggr=aggregator))
        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
          # Construct the network as shown in the model architecture.
          x = self.convs[i](x, edge_index)
          x = self.norms[i](x)
          x = F.relu(x)
          # By setting training to self.training, we will only apply dropout
          # during model training.
          x = F.dropout(x, p = self.dropout)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index)

        return x

class GATConv_v2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout):
        super(GATConv_v2, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        conv_layers = [GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)]
        conv_layers += [GATv2Conv((-1, -1), hidden_channels, add_self_loops=False) for _ in
                        range(num_layers - 2)]
        conv_layers.append(GATv2Conv((-1, -1), out_channels, add_self_loops=False))
        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
          # Construct the network as shown in the model architecture.
          x = self.convs[i](x, edge_index)
          x = self.norms[i](x)
          x = F.relu(x)
          # By setting training to self.training, we will only apply dropout
          # during model training.
          x = F.dropout(x, p = self.dropout)

        x = self.convs[-1](x, edge_index)

        return x