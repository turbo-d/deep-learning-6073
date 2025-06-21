import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(ResidualBlock, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(out_channels)
    )
    self.relu = nn.ReLU()
    self.downsample = downsample
    self.out_channels = out_channels

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet18(nn.Module):
  def __init__(self):
    super(ResNet18, self).__init__()

    self.in_channels = 64

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(ResidualBlock, 64, 2, 1)
    self.layer2 = self._make_layer(ResidualBlock, 128, 2, 2)
    self.layer3 = self._make_layer(ResidualBlock, 256, 2, 2)
    self.layer4 = self._make_layer(ResidualBlock, 512, 2, 2)
    self.fc = nn.Sequential(
      nn.Linear(512, 10),
      nn.Softmax(dim=1)
    )
  
  def _make_layer(self, block, out_channels, n_blocks, stride):
    downsample = None
    if stride != 1:
      downsample = nn.Sequential(
        nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    
    layers = []
    layers.append(block(self.in_channels, out_channels, stride, downsample))

    self.in_channels = out_channels

    for i in range(1, n_blocks):
      layers.append(block(self.in_channels, out_channels))
    
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.max_pool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

  def visualize(self, x):
    x = self.conv1(x)
    x = self.max_pool(x)
    x = self.layer1(x)
    layer1_feats = x
    x = self.layer2(x)
    layer2_feats = x
    return layer1_feats, layer2_feats