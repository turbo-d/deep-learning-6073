from segmentation_dataset import SegmentationDataset
import torch

train_ds = SegmentationDataset("./data/train/")
test_ds = SegmentationDataset("./data/test/")

print(f"Number of samples in Train dataset: {len(train_ds)}")
print(f"Number of samples in Test dataset: {len(test_ds)}")

train_dataloader = torch.utils.data.DataLoader(train_ds)
test_dataloader = torch.utils.data.DataLoader(test_ds)

# Display image and label.
#train_features, train_labels = next(iter(train_dataloader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#img = train_features[0].squeeze()
#label = train_labels[0]
#plt.imshow(img, cmap="gray")
#plt.show()
#print(f"Label: {label}")

for input, _ in train_dataloader:
  print(input.dtype)
  print(input.shape)
  print(f"Max element: {torch.max(input)}")
  print(f"Min element: {torch.min(input)}")
  #print(input)

for _, label in train_dataloader:
  print(label.shape)
  #print(label)