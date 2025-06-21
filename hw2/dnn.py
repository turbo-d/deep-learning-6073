import torch.nn as nn

class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()

    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(784, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 10),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    return self.model(x)

  def visualize(self, x):
    return None, None