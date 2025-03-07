import copy
import torch
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


class UniMP(torch.nn.Module):
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
        super(UniMP, self).__init__()

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
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index)

        return x

def train(model, data, train_idx, optimizer, loss_fn):
    """
    Param:
    - model: The neural network model to be trained.
    - data: The graph data, which includes node features (`data.x`) and edge
        indices (`data.edge_index`).
    - train_idx: Indices of nodes in the training set.
    - optimizer: The optimizer used for updating model parameters.
    - loss_fn: The loss function used to compute the training loss.
    """
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], torch.flatten(data.y[train_idx]))

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx):
    """
    Compute the model accuracy for train, validation, and test dataset.
    Params.
    - model: The neural network model to be evaluated.
    - data: The graph data, containing node features (`data.x`),
        labels (`data.y`), and edge indices (`data.edge_index`).
    - split_idx: A dictionary containing indices for the training, validation,
        and test sets.
    """
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred_train = out[split_idx['train']].argmax(dim=-1, keepdim=False)
    train_acc = int((y_pred_train == data.y[split_idx['train']]).sum()) / y_pred_train.size(0)

    y_pred_val = out[split_idx['valid']].argmax(dim=-1, keepdim=False)
    valid_acc = int((y_pred_val == data.y[split_idx['valid']]).sum()) / y_pred_val.size(0)

    y_pred_test = out[split_idx['test']].argmax(dim=-1, keepdim=False)
    test_acc = int((y_pred_test == data.y[split_idx['test']]).sum()) / y_pred_test.size(0)

    return train_acc, valid_acc, test_acc

dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToUndirected())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = dataset[0]
print(data)
data = data.to(device)
data.y = data.y.view(-1) # To avoid CUDA out of memory during evaluations.

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

configs = {
      'device': device,
      'num_layers': 3,
      # The paper use 128 hidden dim, but we use 64 here due to resourse limitation.
      # Similar issues were found in https://github.com/pyg-team/pytorch_geometric/discussions/3388
      # We found this as a limitation of using transformer layer.
      'hidden_dim': 64,
      'num_heads': 2,
      'dropout': 0.3,
      'lr':  0.001,
      'epochs': 500,
      "weight_decay":0.0005,
  }

model = UniMP(data.num_features, configs['hidden_dim'],
            dataset.num_classes, configs['num_layers'],
            configs['dropout'], heads=configs['num_heads']).to(device)

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
loss_fn = F.cross_entropy

best_model = None
best_valid_acc = 0

losses, train_accs, valid_accs, test_accs = [], [], [], []

epoch_str = 1
for epoch in range(epoch_str, epoch_str + configs["epochs"]):
  loss = train(model, data, train_idx, optimizer, loss_fn)
  result = test(model, data, split_idx)
  train_acc, valid_acc, test_acc = result
  if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      best_model = copy.deepcopy(model)
  losses.append(loss)
  train_accs.append(train_acc)
  valid_accs.append(valid_acc)
  test_accs.append(test_acc)
  if epoch % 10 == 0:
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')