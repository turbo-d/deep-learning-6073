import librosa
from ballroom_dataset import BallroomDataset
from dummy_dataset import DummyBeatDataset
import os

train_set = BallroomDataset(
    audio_dir="./data/BallroomData/",
    annotation_dir="./data/BallroomAnnotations/",
    sample_rate=16000,
    input_length=5,
    hop_length=256,
    time_shrinking=4,
    mode="train"
)
train_set10 = train_set[10]
print(f"Train dataset size: {len(train_set)}")
print(f"audio: {train_set10['audio'].shape}")
print(f"targets: {train_set10['targets'].shape}")
print()

val_set = BallroomDataset(
    audio_dir="./data/BallroomData/",
    annotation_dir="./data/BallroomAnnotations/",
    sample_rate=16000,
    input_length=5,
    hop_length=256,
    time_shrinking=4,
    mode="validation"
)
print(f"Val dataset size: {len(val_set)}")
val_set0 = val_set[0]
print(f"audio: {val_set0['audio'].shape}")
print(f"targets: {val_set0['targets'].shape}")
print(f"beats: {val_set0['beats'].shape}")
print(f"downbeats: {val_set0['downbeats'].shape}")
print()

test_set = BallroomDataset(
    audio_dir="./data/BallroomData/",
    annotation_dir="./data/BallroomAnnotations/",
    sample_rate=16000,
    input_length=5,
    hop_length=256,
    time_shrinking=4,
    mode="test"
)
print(f"Test dataset size: {len(test_set)}")
test_set0 = test_set[0]
print(f"audio: {test_set0['audio'].shape}")
print(f"targets: {test_set0['targets'].shape}")
print(f"beats: {test_set0['beats'].shape}")
print(f"downbeats: {test_set0['downbeats'].shape}")
#print(test_set0["targets"][500:700, :])
#print(test_set0["beats"])
#print(test_set0["downbeats"])
print()

print()

dummy_set_train = DummyBeatDataset(
    sample_rate=16000,
    input_length=5,
    hop_length=256,
    time_shrinking=4,
    mode="train"
)
print(f"Dummy train dataset size: {len(dummy_set_train)}")
print(f"audio: {dummy_set_train[0]['audio'].shape}")
print(f"targets: {dummy_set_train[0]['targets'].shape}")
print()


dummy_set_val = DummyBeatDataset(
    sample_rate=16000,
    input_length=5,
    hop_length=256,
    time_shrinking=4,
    mode="validation"
)
print(f"Dummy val dataset size: {len(dummy_set_val)}")
print(f"audio: {dummy_set_val[0]['audio'].shape}")
print(f"targets: {dummy_set_val[0]['targets'].shape}")
print(f"beats: {dummy_set_val[0]['beats'].shape}")
print(f"downbeats: {dummy_set_val[0]['downbeats'].shape}")
