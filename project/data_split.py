from glob import glob
import torch as th
import os

audio_dir = "./data/BallroomData/"

audio_files = glob(os.path.join(audio_dir, "**", "*.wav"))
print(f"# of Audio files: {len(audio_files)}")

g = th.Generator().manual_seed(2147483647)

# test split
test_size = int(len(audio_files) * 0.1)
rp = th.randperm(len(audio_files), generator=g).tolist()
temp_audio_files = [audio_files[i] for i in rp[:-test_size]]
test_audio_files = [audio_files[i] for i in rp[-test_size:]]
print(f"# of Test files: {len(test_audio_files)}")

# train / val split
val_size = int(len(temp_audio_files) * 0.11111)
rp = th.randperm(len(temp_audio_files), generator=g).tolist()
train_audio_files = [temp_audio_files[i] for i in rp[:-val_size]]
print(f"# of Train files: {len(train_audio_files)}")
val_audio_files = [temp_audio_files[i] for i in rp[-val_size:]]
print(f"# of Val files: {len(val_audio_files)}")

# train
train_file_path = os.path.join(audio_dir, "train.txt")
with open(train_file_path, "w") as f:
    for file in train_audio_files:
        f.write(f"{os.path.relpath(file, start=audio_dir)}\n")

# val
val_file_path = os.path.join(audio_dir, "val.txt")
with open(val_file_path, "w") as f:
    for file in val_audio_files:
        f.write(f"{os.path.relpath(file, start=audio_dir)}\n")

# test
test_file_path = os.path.join(audio_dir, "test.txt")
with open(test_file_path, "w") as f:
    for file in test_audio_files:
        f.write(f"{os.path.relpath(file, start=audio_dir)}\n")