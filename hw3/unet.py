# Implementation inspired by https://towardsdatascience.com/understanding-u-net-61276b10f360

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=False):
    super(ConvLayer, self).__init__()
    self.do_batch_norm = batch_norm

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    if self.do_batch_norm:
      x = self.batch_norm(x)
    return self.relu(x)


class Encoder(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=False):
    super(Encoder, self).__init__()

    self.double_conv = nn.Sequential(
      ConvLayer(in_channels, out_channels, batch_norm=batch_norm),
      ConvLayer(out_channels, out_channels, batch_norm=batch_norm)
    )

    self.max_pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.double_conv(x)
    mp = self.max_pool(x)
    return mp, x


class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=False):
    super(Decoder, self).__init__()

    self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    self.double_conv = nn.Sequential(
      ConvLayer(in_channels, out_channels, batch_norm=batch_norm),
      ConvLayer(out_channels, out_channels, batch_norm=batch_norm),
    )
  
  def forward(self, x, skip):
    x = self.up_conv(x)
    x = torch.cat([skip, x], dim=1)
    x = self.double_conv(x)
    return x
    

class UNet(nn.Module):
  def __init__(self, in_channels, n_classes, layers, batch_norm=False):
    super(UNet, self).__init__()
    self.encoders = nn.ModuleList()
    self.decoders = nn.ModuleList()

    out_channels = 64
    for _ in range(layers+1):
      self.encoders.append(Encoder(in_channels, out_channels, batch_norm=batch_norm))
      in_channels, out_channels = out_channels, out_channels * 2

    out_channels = in_channels // 2
    for _ in range(layers):
      self.decoders.append(Decoder(in_channels, out_channels, batch_norm=batch_norm))
      in_channels, out_channels = out_channels, out_channels // 2

    self.logits = nn.Conv2d(in_channels, n_classes, kernel_size=1)

  def forward(self, x):
    skip_conns = []
    for i, enc in enumerate(self.encoders):
      x, skip = enc(x)
      skip_conns.append(skip)

    x = skip_conns.pop()
    for i, dec in enumerate(self.decoders):
      skip_conn = skip_conns.pop()
      x = dec(x, skip_conn)

    logits = self.logits(x)
    return logits