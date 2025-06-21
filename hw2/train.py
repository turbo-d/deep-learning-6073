import argparse
from dnn import DNN
from lenet import LeNet
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from resnet import ResNet18
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchvision import datasets, transforms

def train(model, loss_fn, optimizer, device, train_dataloader):
  model.train()

  start_time = time.time()
  train_loss = 0
  correct = 0
  for inputs, labels in train_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    # Backprop
    loss.backward()

    # Update weights
    optimizer.step()

    # Training batch metrics
    train_loss += loss.item()
    labels_class = torch.argmax(labels, dim=1)
    outputs_class = torch.argmax(outputs, dim=1)
    correct += (outputs_class == labels_class).float().sum()
    f1_score = multiclass_f1_score(outputs_class, labels_class, num_classes=10)

  end_time = time.time()

  # Training epoch metrics
  train_loss /= len(train_dataloader)
  elapsed_time = end_time - start_time
  avg_step_time = elapsed_time / len(train_dataloader)
  return train_loss, correct.cpu(), f1_score.cpu(), elapsed_time, avg_step_time

def validate(model, loss_fn, device, val_dataloader):
  model.eval()

  val_loss = 0
  correct = 0
  with torch.no_grad():
    for inputs, labels in val_dataloader:
      inputs, labels = inputs.to(device), labels.to(device)
      
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)

      # Validation batch metrics
      val_loss += loss.item()
      labels_class = torch.argmax(labels, dim=1)
      outputs_class = torch.argmax(outputs, dim=1)
      correct += (outputs_class == labels_class).float().sum()
      f1_score = multiclass_f1_score(outputs_class, labels_class, num_classes=10)

  # Validation epoch metrics
  val_loss /= len(val_dataloader)
  return val_loss, correct.cpu(), f1_score.cpu()

def main(args):
  USE_CUDA = torch.cuda.is_available()
  USE_MPS = torch.backends.mps.is_available()

  DEVICE = torch.device("cpu")
  if USE_CUDA:
    DEVICE = torch.device("cuda")
  elif USE_MPS:
    DEVICE = torch.device("mps")

  # Load model
  model = None
  if args.model == "lenet":
    model = LeNet()
  elif args.model == "dnn":
    model = DNN()
  elif args.model == "resnet":
    model = ResNet18()
  else:
    print("Invalid model selected")
    exit(2)
  model.to(DEVICE)

  # Create directory for model artifacts
  model_name = f"{args.model}_{str(args.lr)}"
  model_dir = f"./{model_name}/"
  Path(model_dir).mkdir(parents=True, exist_ok=True)

  # Load data
  train_kwargs = {"batch_size": args.batch_size}
  if USE_CUDA:
    cuda_kwargs = {"num_workers": 1,
                    "pin_memory": True,
                    "shuffle": True}
    train_kwargs.update(cuda_kwargs)

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
  target_transform = transforms.Compose([
    lambda x: torch.LongTensor([x]),
    lambda x: F.one_hot(x, 10).float(),
    lambda x: torch.squeeze(x)
  ])
  dataset = datasets.MNIST("../data", train=True, download=True,
                      transform=transform, target_transform=target_transform)
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

  train_loader = DataLoader(train_dataset,**train_kwargs)
  val_loader = DataLoader(val_dataset, **train_kwargs)


  # Loss function
  loss_fn = torch.nn.CrossEntropyLoss()

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Training loop
  train_losses = []
  train_accs = []
  train_f1_scores = []
  val_losses = []
  val_accs = []
  val_f1_scores = []
  best_val_loss = sys.float_info.max
  with open(f"{model_dir}training.log", "w") as f:
    for epoch in range(1, args.epochs + 1):
      train_loss, n_correct_train_preds, train_f1_score, epoch_train_time, avg_step_train_time = train(model, loss_fn, optimizer, DEVICE, train_loader)
      train_losses.append(train_loss)
      train_acc = 100 * n_correct_train_preds / len(train_dataset)
      train_accs.append(train_acc.numpy())
      train_f1_scores.append(train_f1_score.numpy())

      val_loss, n_correct_val_preds, val_f1_score = validate(model, loss_fn, DEVICE, val_loader)
      val_losses.append(val_loss)
      val_acc = 100 * n_correct_val_preds / len(val_dataset)
      val_accs.append(val_acc.numpy())
      val_f1_scores.append(val_f1_score.numpy())

      # Save model with early stopping
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = f"{model_dir}model"
        torch.save(model.state_dict(), model_path)

      # Log epoch stats
      epoch_stats = f"{epoch}/{args.epochs} - {epoch_train_time:.0f}s - train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - train acc: {train_acc:.2f} - val acc: {val_acc:.2f} - train f1: {train_f1_score:.2f} - val f1: {val_f1_score:.2f} - {avg_step_train_time:.4f}s/step"
      print(epoch_stats)
      print(epoch_stats, file=f)


  # Plots
  xs = np.arange(1, args.epochs+1)

  # Loss plot
  fig, ax = plt.subplots()
  ax.plot(xs, train_losses)
  ax.plot(xs, val_losses)
  ax.set_title(f"{model_name} Train/Val Loss")
  ax.set_ylabel("loss")
  ax.legend(["train", "val"], loc="upper right")
  fig.savefig(f"{model_dir}train_val_loss.png")


  # Metrics plot
  fig, axs = plt.subplots(1, 2, sharex=True)
  fig.suptitle(f"{model_name} Train/Val Metrics")
  fig.supxlabel("epoch")

  # Accuracy plot
  axs[0].plot(xs, train_accs)
  axs[0].plot(xs, val_accs)
  axs[0].set_title("accuracy")
  axs[0].set_ylabel("accuracy")
  axs[0].legend(["train", "val"], loc="lower right")

  # F1 Score plot
  axs[1].plot(xs, train_f1_scores)
  axs[1].plot(xs, val_f1_scores)
  axs[1].set_title("f1 score")
  axs[1].set_ylabel("f1 score")
  axs[1].legend(["train", "val"], loc="lower right")

  fig.set_figwidth(12)
  fig.set_figheight(6)
  fig.tight_layout()
  fig.savefig(f"{model_dir}train_val_metrics.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", choices=["lenet", "dnn", "resnet"], type=str.lower)
  parser.add_argument("--lr", default=1e-2, type=float)
  parser.add_argument("--epochs", default=10, type=int)
  parser.add_argument("--batch_size", default=128, type=int)
  args = parser.parse_args()
  main(args)