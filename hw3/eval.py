import argparse
import monai
from segmentation_dataset import SegmentationDataset
import torch
from unet import UNet

logits_to_mask = monai.transforms.Compose([
  monai.transforms.Activations(sigmoid=True),
  monai.transforms.AsDiscrete(threshold=0.5),
])

def main(args):
  # Setup model path
  model_name = f"unet_{args.layers}_layer"
  if args.batch_norm:
    model_name += "_batch_norm"
  model_dir = f"./{model_name}/"
  model_path = model_dir + "model"

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
  test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin_memory)

  # Load model
  in_channels = 3
  out_channels = 1
  model = UNet(in_channels, out_channels, args.layers, args.batch_norm)
  model.load_state_dict(torch.load(model_path))
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  # Metrics
  loss_function = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
  dice_metric = monai.metrics.DiceMetric()
  iou_metric = monai.metrics.MeanIoU()

  # Eval loop
  test_loss = 0
  n_test_steps = 0
  dice_metric.reset()
  iou_metric.reset()

  model.eval()
  with torch.no_grad():
    for inputs, labels in test_dataloader:
      n_test_steps += 1
      
      outputs = model(inputs)

      loss = loss_function(outputs, labels)
      test_loss += loss.item()

      mask_preds = logits_to_mask(outputs)
      dice_metric(y_pred=mask_preds, y=labels)
      iou_metric(y_pred=mask_preds, y=labels)
  
  # Test metrics
  test_loss /= n_test_steps
  test_dice = dice_metric.aggregate().item()
  test_iou = iou_metric.aggregate().item()

  # Display metrics
  print(model_name)
  print(f"loss: {test_loss:.4f} - dice: {test_dice:.4f} - iou: {test_iou:.4f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-norm", dest="batch_norm", action="store_true")
  parser.add_argument("--batch-size", dest="batch_size", default=2, type=int)
  parser.add_argument("--num-layers", dest="layers", default=2, type=int)
  args = parser.parse_args()
  main(args)