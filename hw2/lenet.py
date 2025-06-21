import torch.nn as nn

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 6, kernel_size=5, stride=1, padding="same"),
      nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5, stride=1),
      nn.ReLU(),
    )
    self.conv3 = nn.Sequential(
      nn.Conv2d(16, 120, kernel_size=5, stride=1),
      nn.ReLU(),
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, 10),
      nn.Softmax(dim=1)
    )
    self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.avg_pool(x)
    x = self.conv2(x)
    x = self.avg_pool(x)
    x = self.conv3(x)
    x = self.fc(x)
    return x

  def visualize(self, x):
    x = self.conv1(x)
    layer1_feats = x
    x = self.avg_pool(x)
    x = self.conv2(x)
    layer2_feats = x
    return layer1_feats, layer2_feats