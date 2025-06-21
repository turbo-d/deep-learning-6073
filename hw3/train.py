import argparse
import matplotlib.pyplot as plt
import monai
import numpy as np
from pathlib import Path
from segmentation_dataset import SegmentationDataset
import sys
import time
import torch
from unet import UNet

logits_to_mask = monai.transforms.Compose([
  monai.transforms.Activations(sigmoid=True),
  monai.transforms.AsDiscrete(threshold=0.5),
])

def main(args):
  # Create directory for model artifacts
  model_name = f"unet_{args.layers}_layer"
  if args.batch_norm:
    model_name += "_batch_norm"
  model_dir = f"./{model_name}/"
  Path(model_dir).mkdir(parents=True, exist_ok=True)

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
  train_ds = SegmentationDataset("./data/train/", device=device)
  train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.75, 0.25])
  train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin_memory)
  val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin_memory)

  # Load model
  in_channels = 3
  out_channels = 1
  model = UNet(in_channels, out_channels, args.layers, args.batch_norm)
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  # Loss function
  loss_function = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), args.lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

  # Metrics
  dice_metric = monai.metrics.DiceMetric()
  iou_metric = monai.metrics.MeanIoU()

  # Training loop
  train_losses = []
  train_dices = []
  train_ious = []
  val_losses = []
  val_dices = []
  val_ious = []
  best_val_loss = sys.float_info.max
  with open(model_dir + "training.log", "w") as f:
    for epoch in range(args.epochs):
      epoch_counter = f"Epoch {epoch+1}/{args.epochs}"
      print(epoch_counter)
      print(epoch_counter, file=f)

      # Mini-batch training
      start_time = time.time()
      model.train()
      train_loss = 0
      n_train_steps = 0
      dice_metric.reset()
      iou_metric.reset()
      for inputs, labels in train_dataloader:
        print(f"Step {n_train_steps+1}", end="", flush=True)
        batch_start_time = time.time()
        n_train_steps += 1

        # Forward pass
        print(f" - forward: ", end="", flush=True)
        start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        print(f"{(time.time() - start):.3f}s", end="", flush=True)

        # Training batch metrics
        train_loss += loss.item()
        mask_preds = logits_to_mask(outputs)
        dice_metric(y_pred=mask_preds, y=labels)
        iou_metric(y_pred=mask_preds, y=labels)

        # Backprop
        print(f" - backprop: ", end="", flush=True)
        start = time.time()
        loss.backward()
        print(f"{(time.time() - start):.3f}s", end="", flush=True)

        # Update weights
        print(f" - update: ", end="", flush=True)
        start = time.time()
        optimizer.step()
        print(f"{(time.time() - start):.3f}s", end="", flush=True)

        print(f" - total: {(time.time() - batch_start_time):.3f}s")

      # Training epoch metrics
      epoch_train_time = time.time() - start_time
      avg_train_step_time = epoch_train_time / n_train_steps
      train_loss /= n_train_steps
      train_losses.append(train_loss)
      train_dice = dice_metric.aggregate().item()
      train_dices.append(train_dice)
      train_iou = iou_metric.aggregate().item()
      train_ious.append(train_iou)

      # Calculate epoch validation metrics
      model.eval()
      val_loss = 0
      n_val_steps = 0
      dice_metric.reset()
      iou_metric.reset()
      with torch.no_grad():
        for inputs, labels in val_dataloader:
          n_val_steps += 1
          
          outputs = model(inputs)

          loss = loss_function(outputs, labels)
          val_loss += loss.item()

          mask_preds = logits_to_mask(outputs)
          dice_metric(y_pred=mask_preds, y=labels)
          iou_metric(y_pred=mask_preds, y=labels)
      
      # Validation epoch metrics
      val_loss /= n_val_steps
      val_losses.append(val_loss)
      val_dice = dice_metric.aggregate().item()
      val_dices.append(val_dice)
      val_iou = iou_metric.aggregate().item()
      val_ious.append(val_iou)

      # Save model with early stopping
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = model_dir + "model"
        torch.save(model.state_dict(), model_path)

      # Update learning rate
      scheduler.step(val_loss)
      
      # Log epoch stats
      epoch_stats = f"{n_train_steps}/{n_train_steps} - {epoch_train_time:.0f}s - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - {epoch_train_time:.4f}s/epoch - {avg_train_step_time:.4f}s/step"
      print(epoch_stats)
      print(epoch_stats, file=f)

  # Plots
  xs = np.arange(1, args.epochs+1)
  fig, axs = plt.subplots(1, 3, sharex=True)
  fig.suptitle(model_name)
  fig.supxlabel('epoch')

  # Loss plot
  axs[0].plot(xs, train_losses)
  axs[0].plot(xs, val_losses)
  axs[0].set_title('loss')
  axs[0].set_ylabel('loss')
  axs[0].legend(['train', 'val'], loc='upper right')

  # Dice plot
  axs[1].plot(xs, train_dices)
  axs[1].plot(xs, val_dices)
  axs[1].set_title('dice')
  axs[1].set_ylabel('dice')
  axs[1].legend(['train', 'val'], loc='lower right')

  # IoU plot
  axs[2].plot(xs, train_ious)
  axs[2].plot(xs, val_ious)
  axs[2].set_title('IoU')
  axs[2].set_ylabel('IoU')
  axs[2].legend(['train', 'val'], loc='lower right')

  fig.set_figwidth(18)
  fig.set_figheight(6)
  fig.tight_layout()
  fig.savefig(model_dir + 'train_val_plots.png')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-norm", dest="batch_norm", action="store_true")
  parser.add_argument("--batch-size", dest="batch_size", default=8, type=int)
  parser.add_argument("--epochs", default=100, type=int)
  parser.add_argument("--lr", default=1e-2, type=float)
  parser.add_argument("--num-layers", dest="layers", default=2, type=int)
  args = parser.parse_args()
  main(args)