from glob import glob
import os
import torch.utils.data as tud
import pytorch_lightning as pl
import torch as th
import librosa
import torchaudio
from random import shuffle

class BallroomDataset(tud.Dataset):
    def __init__(self, audio_dir, annotation_dir, sample_rate, input_length, hop_length, fft_win_length, pooling_shrinking, mode):
        super(BallroomDataset, self).__init__()

        assert mode in ["train", "validation", "test"]
        self.mode = mode

        self.sample_rate = sample_rate
        self.input_length = input_length
        self.hop_length = hop_length
        self.fft_win_length = fft_win_length
        self.pooling_shrinking = pooling_shrinking

        audio_files_index = ""
        if self.mode == "train":
            audio_files_index = os.path.join(audio_dir, "train.txt")
        elif self.mode == "validation":
            audio_files_index = os.path.join(audio_dir, "val.txt")
        elif self.mode == "test":
            audio_files_index = os.path.join(audio_dir, "test.txt")

        # get audio files for given mode
        self.audio_files = []
        with open(audio_files_index, "r") as f:
            for audio_file in f:
                audio_file = audio_file.strip()
                self.audio_files.append(os.path.join(audio_dir, audio_file))

        # get annotation files for given mode
        self.ann_files = []
        for audio_file in self.audio_files:
            self.ann_files.append(os.path.join(annotation_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".beats"))
        
        # get audio chunks for given mode
        self.audio_chunks = [] # ('index into self.audio_files', 'offset in seconds from start of audio file')
        if self.mode == "train":
            for i, audio_file in enumerate(self.audio_files):
                sample_rate = librosa.get_samplerate(audio_file)
                duration = librosa.get_duration(path=audio_file, sr=sample_rate)
                offsets = th.arange(0, int(duration - self.input_length) + 1)
                for offset in offsets:
                    self.audio_chunks.append((i, offset))
            shuffle(self.audio_chunks)
        else:
            for i, _ in enumerate(self.audio_files):
                self.audio_chunks.append((i, 0))
        
    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self, i):
        idx, chunk_offset_in_seconds = self.audio_chunks[i]

        # load audio data
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        nsamples = tuple(waveform.shape)[1]
        waveform = waveform.flatten()
        #duration = nsamples / sample_rate

        # load annotation data
        ann_file = self.ann_files[idx]
        beat_offsets_in_seconds = []
        beats = []
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                beat_offsets_in_seconds.append(float(line[0]))
                beats.append(int(line[1]))

        # generate targets
        nframes = int((1 + (nsamples - self.fft_win_length) // self.hop_length) // self.pooling_shrinking)
        samples_per_frame = nsamples / nframes

        targets = [[0,0,1] for i in range(nframes)] # init to 'non-beat'
        for i, beat_offset_in_seconds in enumerate(beat_offsets_in_seconds):
            beat_offset_in_samples = beat_offset_in_seconds * self.sample_rate
            beat_offset_in_frames = int(beat_offset_in_samples // samples_per_frame)
            if beat_offset_in_frames < len(targets):
                if beats[i] == 1: # downbeat
                    targets[beat_offset_in_frames] = [0,1,0] # set to 'downbeat'
                else: # beat
                    targets[beat_offset_in_frames] = [1,0,0] # set to 'beat'

        targets = th.Tensor(targets)

        # return mode-specific data
        if self.mode == "train":
            # compute audio chunk
            chunk_length_in_samples = self.input_length * self.sample_rate
            chunk_offset_in_samples = chunk_offset_in_seconds * self.sample_rate
            audio_chunk = waveform[chunk_offset_in_samples:chunk_offset_in_samples + chunk_length_in_samples]

            # compute targets chunk
            chunk_length_in_frames = int((1 + (chunk_length_in_samples - self.fft_win_length) // self.hop_length) // self.pooling_shrinking)
            chunk_offset_in_frames = int(chunk_offset_in_samples // samples_per_frame)
            targets_chunk = targets[chunk_offset_in_frames:chunk_offset_in_frames + chunk_length_in_frames, :]

            return {
                'audio': audio_chunk,
                'targets': targets_chunk
            }
        elif self.mode in ["validation", "test"]:
            # get downbeat offsets
            downbeat_offsets_in_seconds = []
            for i, beat in enumerate(beats):
                if beat == 1:
                    downbeat_offsets_in_seconds.append(beat_offsets_in_seconds[i])

            return {
                'audio': waveform,
                'targets': targets,
                'beats': th.Tensor(beat_offsets_in_seconds),
                'downbeats': th.Tensor(downbeat_offsets_in_seconds)
            }

class BallroomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, pin_memory, audio_dir, annotation_dir, sample_rate, input_length, hop_length, fft_win_length, pooling_shrinking):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.sample_rate = sample_rate
        self.input_length = input_length
        self.hop_length = hop_length
        self.fft_win_length = fft_win_length
        self.pooling_shrinking = pooling_shrinking

    def setup(self, stage):
        self.train_set = BallroomDataset(
            audio_dir=self.audio_dir,
            annotation_dir=self.annotation_dir,
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            hop_length=self.hop_length,
            fft_win_length=self.fft_win_length,
            pooling_shrinking=self.pooling_shrinking,
            mode="train"
        )
        self.val_set = BallroomDataset(
            audio_dir=self.audio_dir,
            annotation_dir=self.annotation_dir,
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            hop_length=self.hop_length,
            fft_win_length=self.fft_win_length,
            pooling_shrinking=self.pooling_shrinking,
            mode="validation"
        )
        self.test_set = BallroomDataset(
            audio_dir=self.audio_dir,
            annotation_dir=self.annotation_dir,
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            hop_length=self.hop_length,
            fft_win_length=self.fft_win_length,
            pooling_shrinking=self.pooling_shrinking,
            mode="test"
        )

    def prepare_data_per_node(self):
        return None

    def train_dataloader(self):
        return tud.DataLoader(self.train_set,
                              batch_size=self.batch_size,
                              pin_memory=self.pin_memory,
                              shuffle=True,
                              num_workers=self.n_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.val_set,
                              batch_size=1,
                              pin_memory=self.pin_memory,
                              shuffle=False,
                              num_workers=self.n_workers)

    def test_dataloader(self):
        return tud.DataLoader(self.test_set,
                              batch_size=1,
                              pin_memory=self.pin_memory,
                              shuffle=False,
                              num_workers=self.n_workers)