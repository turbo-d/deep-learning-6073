from glob import glob
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class SegmentationDataset(Dataset):
  def __init__(self, img_dir, transform=None, target_transform=None, device="cpu"):
    super(SegmentationDataset, self).__init__()
    self.transform = transform
    self.target_transform = target_transform
    self.device = device

    self.images = sorted(glob(os.path.join(img_dir, "image", "*.png")))
    self.masks = sorted(glob(os.path.join(img_dir, "mask", "*.png")))

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path = self.images[idx]
    img = read_image(img_path)
    img = img.float()
    if self.device != "cpu":
      img = img.to(self.device)
    if self.transform is not None:
      img = self.transform(img)

    mask_path = self.masks[idx]
    mask = read_image(mask_path)
    mask = mask.float()
    if self.device != "cpu":
      mask = mask.to(self.device)
    if self.target_transform is not None:
      mask = self.target_transform(mask)

    return img, mask