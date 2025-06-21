import argparse
from dnn import DNN
from lenet import LeNet
import matplotlib.pyplot as plt
from resnet import ResNet18
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchvision import datasets, transforms

def test(model, loss_fn, device, test_dataloader):
  model.eval()

  test_loss = 0
  correct = 0
  with torch.no_grad():
    for inputs, labels in test_dataloader:
      inputs, labels = inputs.to(device), labels.to(device)
      
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)

      # Validation batch metrics
      test_loss += loss.item()
      labels_class = torch.argmax(labels, dim=1)
      outputs_class = torch.argmax(outputs, dim=1)
      correct += (outputs_class == labels_class).float().sum()
      f1_score = multiclass_f1_score(outputs_class, labels_class, num_classes=10)
      fprs, tprs, _ = roc_curve(torch.ravel(labels).cpu(), torch.ravel(outputs).cpu())

  # Validation epoch metrics
  test_loss /= len(test_dataloader)
  return test_loss, correct.cpu(), f1_score.cpu(), fprs, tprs

def feature_visualization(input, model, device, model_name, model_dir):
  model.eval()
  with torch.no_grad():
    input = input.to(device)
    layer1, layer2 = model.visualize(input)
    if layer1 == None or layer2 == None:
      return
    
    #print("feat vis:")
    #print(f"input: {input.shape}")
    #print(f"layer1: {layer1.shape}")
    #print(f"layer2: {layer2.shape}")
    #print(f"layer1[0]: {layer1[0].shape}")

    # Plot features
    input = input.cpu().numpy()
    layer1 = layer1.cpu().numpy()
    layer2 = layer2.cpu().numpy()
    for i, layer in enumerate([layer1, layer2]):
      fig, axs = plt.subplots(1, 5)
      fig.suptitle(f"{model_name} Layer {i+1} Features")
      
      # Input
      axs[0].imshow(input.squeeze())
      axs[0].axis("off")
      axs[0].set_title("input")

      # Show first 4 feature maps of layer
      for j in range(4):
        axs[j+1].imshow(layer.squeeze()[j,:,:])
        axs[j+1].axis("off")
        axs[j+1].set_title(f"feat {j}")

      fig.set_figwidth(18)
      fig.set_figheight(6)
      fig.tight_layout()
      fig.savefig(f"{model_dir}layer{i+1}_feats.png")

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

  model_name = f"{args.model}_{str(args.lr)}"
  model_dir = f"./{model_name}/"

  # Load trained model
  model_path = model_dir + "model"
  model.load_state_dict(torch.load(model_path, map_location=DEVICE))
  model.to(DEVICE)

  # Load data
  test_kwargs = {"batch_size": args.batch_size}
  if USE_CUDA:
    cuda_kwargs = {"num_workers": 1,
                    "pin_memory": True,
                    "shuffle": True}
    test_kwargs.update(cuda_kwargs)

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
  target_transform = transforms.Compose([
    lambda x: torch.LongTensor([x]),
    lambda x: F.one_hot(x, 10).float(),
    lambda x: torch.squeeze(x)
  ])
  test_dataset = datasets.MNIST("../data", train=False,
                      transform=transform, target_transform=target_transform)
  test_loader = DataLoader(test_dataset, **test_kwargs)

  # Test model
  loss_fn = torch.nn.CrossEntropyLoss()
  test_loss, n_correct_test_preds, test_f1_score, fprs, tprs = test(model, loss_fn, DEVICE, test_loader)
  test_acc = 100 * n_correct_test_preds / len(test_dataset)
  test_roc_auc = auc(fprs, tprs)

  print(f"{model_name}:")
  print(f"test loss: {test_loss:.4f}")
  print(f"test acc: {test_acc:.2f}")
  print(f"test f1: {test_f1_score:.2f}")
  print(f"test ROC AUC: {test_roc_auc:.2f}")

  # Plot ROC curve
  fig, ax = plt.subplots()
  ax.plot(fprs, tprs, label="AUC = {:.3f}".format(test_roc_auc))
  ax.set_xlabel("False Positive Rate (FPR)")
  ax.set_ylabel("True Positive Rate (TPR)")
  ax.set_title(f"{model_name} Micro-Averaged OvR ROC curve")
  ax.legend(loc="lower right")
  # Plot random performance line for reference
  ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
  fig.savefig(f"{model_dir}roc_curve.png")

  # Feature visualization
  feature_visualization(torch.unsqueeze(test_dataset[0][0], 0), model, DEVICE, model_name, model_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", choices=["lenet", "dnn", "resnet"], type=str.lower)
  parser.add_argument("--lr", default=1e-2, type=float)
  parser.add_argument("--batch_size", default=128, type=int)
  args = parser.parse_args()
  main(args)