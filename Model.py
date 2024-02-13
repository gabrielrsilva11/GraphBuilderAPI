from torch_geometric.nn import GATConv, Linear, SAGEConv, SplineConv
import torch
import torch.nn.functional as F


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

class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, hidden_channels, out_channels):
    super().__init__()
    self.sage1 = SAGEConv((-1, -1), hidden_channels)
    self.sage2 = SAGEConv((-1, -1), 64)
    self.sage3 = SAGEConv((-1, -1), 32)
    self.sage4 = SAGEConv((-1, -1), out_channels)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage3(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage4(h, edge_index)
    return F.log_softmax(h, dim=1)

  def fit(self, train_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0

      # Train on batches
      for batch in train_loader:
        optimizer.zero_grad()
        _, out = self(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        total_loss += loss
        acc += accuracy(out[batch.train_mask].argmax(dim=1),
                        batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
        val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                            batch.y[batch.val_mask])

      # Print metrics every 10 epochs
      if(epoch % 100 == 0):
          print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} '
                f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                f'{val_loss/len(train_loader):.2f} | Val Acc: '
                f'{val_acc/len(train_loader)*100:.2f}%')