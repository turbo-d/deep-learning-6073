import matplotlib.pyplot as plt
import monai
from segmentation_dataset import SegmentationDataset
import torch
from unet import UNet

logits_to_mask = monai.transforms.Compose([
  monai.transforms.Activations(sigmoid=True),
  monai.transforms.AsDiscrete(threshold=0.5),
])

# Use GPU acceleration if available
device = "cpu"
use_pin_memory = False
if torch.backends.mps.is_available():
  device = "mps"
  #use_pin_memory = True
elif torch.cuda.is_available():
  device = "cuda"
  #use_pin_memory = True

# Load data
test_ds = SegmentationDataset("./data/test/", device=device)
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=True, pin_memory=use_pin_memory)

in_channels = 3
out_channels = 1

# Load best 2 layer model
model2_name = f"unet_2_layer"
model2_dir = f"./{model2_name}/"
model2_path = model2_dir + "model"
model2 = UNet(in_channels, out_channels, 2, batch_norm=False)
model2.load_state_dict(torch.load(model2_path))
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
  model2 = torch.nn.DataParallel(model2)
model2.to(device)

# Load best 3 layer model
model3_name = f"unet_3_layer"
model3_dir = f"./{model3_name}/"
model3_path = model3_dir + "model"
model3 = UNet(in_channels, out_channels, 3, batch_norm=False)
model3.load_state_dict(torch.load(model3_path))
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
  model3 = torch.nn.DataParallel(model3)
model3.to(device)

# Load best 4 layer model
model4_name = f"unet_4_layer"
model4_dir = f"./{model4_name}/"
model4_path = model4_dir + "model"
model4 = UNet(in_channels, out_channels, 4, batch_norm=False)
model4.load_state_dict(torch.load(model4_path))
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
  model4 = torch.nn.DataParallel(model4)
model4.to(device)

model2.eval()
model3.eval()
model4.eval()
with torch.no_grad():
  for inputs, labels in test_dataloader:
    outputs2 = model2(inputs)
    mask_preds2 = logits_to_mask(outputs2)

    outputs3 = model3(inputs)
    mask_preds3 = logits_to_mask(outputs3)

    outputs4 = model3(inputs)
    mask_preds4 = logits_to_mask(outputs4)

    inputs = inputs.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    mask_preds2 = mask_preds2.detach().cpu().clone().numpy()
    mask_preds3 = mask_preds3.detach().cpu().clone().numpy()
    mask_preds4 = mask_preds4.detach().cpu().clone().numpy()

    # Create prediction image
    fig, axs = plt.subplots(4, 5)
    
    for row in range(4):
        # Input
        axs[row, 0].imshow(inputs[row].T)
        axs[row, 0].axis('off')

        # Label
        axs[row, 1].imshow(labels[row].T)
        axs[row, 1].axis('off')

        # 2 layer
        axs[row, 2].imshow(mask_preds2[row].T)
        axs[row, 2].axis('off')

        # 3 layer
        axs[row, 3].imshow(mask_preds3[row].T)
        axs[row, 3].axis('off')

        # 4 layer
        axs[row, 4].imshow(mask_preds4[row].T)
        axs[row, 4].axis('off')

    fig.set_figwidth(18)
    fig.set_figheight(6)
    fig.tight_layout()
    fig.savefig('example_predictions.png')

    # Only need one batch of size 4
    break
