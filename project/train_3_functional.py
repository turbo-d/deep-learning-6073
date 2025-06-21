#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytorch_lightning as pl
import librosa
import torchaudio
import numpy as np
import torch as th
import torch.nn as nn
import torch.utils.data as tud
import mir_eval


# In[ ]:





# In[ ]:


##HARMONICS TFT BLOCK##

def hz_to_midi(hz):
    return 12 * (th.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))


def note_to_midi(note):
    return librosa.core.note_to_midi(note)


def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi('C1')
    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)
    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])
    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i+1)))
    return harmonic_hz, level


class HarmonicSTFT(nn.Module):
    """
    Trainable harmonic filters as implemented by Minz Won.
    
    Paper: https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf
    Code: https://github.com/minzwon/data-driven-harmonic-filters
    Pretrained: https://github.com/minzwon/sota-music-tagging-models/tree/master/training
    """

    def __init__(self,
                 sample_rate=16000,
                 n_fft=513,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_Q=1.0,
                 learn_bw=None,
                 checkpoint=None):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                      hop_length=hop_length, pad=pad,
                                                      window_fn=th.hann_window,
                                                      power=power, normalized=normalized, wkwargs=None)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale)

        # Center frequncies to tensor
        self.f0 = th.tensor(harmonic_hz.astype('float32'))

        # Bandwidth parameters
        if learn_bw == 'only_Q':
            self.bw_Q = nn.Parameter(th.tensor(
                np.array([bw_Q]).astype('float32')))
        elif learn_bw == 'fix':
            self.bw_Q = th.tensor(np.array([bw_Q]).astype('float32'))

        if checkpoint is not None:
            state_dict = th.load(checkpoint)
            hstft_state_dict = {k.replace('hstft.', ''): v for k,
                                v in state_dict.items() if 'hstft.' in k}
            self.load_state_dict(hstft_state_dict)

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = th.matmul(fft_bins, (2/bw)) + 1 - (2 * f0 / bw)
        down_slope = th.matmul(fft_bins, (-2/bw)) + 1 + (2 * f0 / bw)
        fb = th.max(self.zero, th.min(down_slope, up_slope))
        return fb

    def to_device(self, device, n_bins):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)
        # fft bins
        self.fft_bins = th.linspace(0, self.sample_rate//2, n_bins)
        self.fft_bins = self.fft_bins.to(device)
        self.zero = th.zeros(1)
        self.zero = self.zero.to(device)

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)
        # to device
        self.to_device(waveform.device, spectrogram.size(1))
        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = th.matmul(
            spectrogram.transpose(1, 2), harmonic_fb).transpose(1, 2)
        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, c, l = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(b, self.n_harmonic, self.level, l)
        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec


# In[ ]:


##NETWORKS BLOCK##

class Res2DMaxPoolModule(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(tuple(pooling))

        # residual
        self.diff = False
        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(
                in_channels, out_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(out_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class ResFrontEnd(nn.Module):
    """
    Adapted from Minz Won ResNet implementation.
    
    Original code: https://github.com/minzwon/semi-supervised-music-tagging-transformer/blob/master/src/modules.py
    """
    def __init__(self, in_channels, out_channels, freq_pooling, time_pooling):
        super(ResFrontEnd, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.layer1 = Res2DMaxPoolModule(
            in_channels, out_channels, pooling=(freq_pooling[0], time_pooling[0]))
        self.layer2 = Res2DMaxPoolModule(
            out_channels, out_channels, pooling=(freq_pooling[1], time_pooling[1]))
        self.layer3 = Res2DMaxPoolModule(
            out_channels, out_channels, pooling=(freq_pooling[2], time_pooling[2]))

    def forward(self, hcqt):
        """
        Inputs:
            hcqt: [B, F, K, T]

        Outputs:
            out: [B, ^F, ^K, ^T]
        """
        # batch normalization
        out = self.input_bn(hcqt)

        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


class SpecTNTBlock(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, dropout, use_tct
    ):
        super().__init__()

        self.D = embed_dim
        self.F = n_frequencies
        self.K = n_channels
        self.T = n_times

        # TCT: Temporal Class Token
        if use_tct:
            self.T += 1

        # Shared frequency-time linear layers
        self.D_to_K = nn.Linear(self.D, self.K)
        self.K_to_D = nn.Linear(self.K, self.D)

        # Spectral Transformer Encoder
        self.spectral_linear_in = nn.Linear(self.F+1, spectral_dmodel)
        self.spectral_encoder_layer = nn.TransformerEncoderLayer(
            d_model=spectral_dmodel, nhead=spectral_nheads, dim_feedforward=spectral_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.spectral_linear_out = nn.Linear(spectral_dmodel, self.F+1)

        # Temporal Transformer Encoder
        self.temporal_linear_in = nn.Linear(self.T, temporal_dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dmodel, nhead=temporal_nheads, dim_feedforward=temporal_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_linear_out = nn.Linear(temporal_dmodel, self.T)

    def forward(self, spec_in, temp_in):
        """
        Inputs:
            spec_in: spectral embedding input [B, T, F+1, K]
            temp_in: temporal embedding input [B, T, 1, D]

        Outputs:
            spec_out: spectral embedding output [B, T, F+1, K]
            temp_out: temporal embedding output [B, T, 1, D]
        """
        # Element-wise addition between TE and FCT
        spec_in = spec_in +             nn.functional.pad(self.D_to_K(temp_in), (0, 0, 0, self.F))

        # Spectral Transformer
        spec_in = spec_in.flatten(0, 1).transpose(1, 2)  # [B*T, K, F+1]
        emb = self.spectral_linear_in(spec_in)  # [B*T, K, spectral_dmodel]
        spec_enc_out = self.spectral_encoder_layer(
            emb)  # [B*T, K, spectral_dmodel]
        spec_out = self.spectral_linear_out(spec_enc_out)  # [B*T, K, F+1]
        spec_out = spec_out.view(-1, self.T, self.K,
                                 self.F+1).transpose(2, 3)  # [B, T, F+1, K]

        # FCT slicing (first raw) + back to D
        temp_in = temp_in + self.K_to_D(spec_out[:, :, :1, :])  # [B, T, 1, D]

        # Temporal Transformer
        temp_in = temp_in.permute(0, 2, 3, 1).flatten(0, 1)  # [B, D, T]
        emb = self.temporal_linear_in(temp_in)  # [B, D, temporal_dmodel]
        temp_enc_out = self.temporal_encoder_layer(
            emb)  # [B, D, temporal_dmodel]
        temp_out = self.temporal_linear_out(temp_enc_out)  # [B, D, T]
        temp_out = temp_out.unsqueeze(1).permute(0, 3, 1, 2)  # [B, T, 1, D]

        return spec_out, temp_out


class SpecTNTModule(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, n_blocks, dropout, use_tct
    ):
        super().__init__()

        D = embed_dim
        F = n_frequencies
        K = n_channels
        T = n_times

        # Frequency Class Token
        self.fct = nn.Parameter(th.zeros(1, T, 1, K))

        # Frequency Positional Encoding
        self.fpe = nn.Parameter(th.zeros(1, 1, F+1, K))

        # TCT: Temporal Class Token
        if use_tct:
            self.tct = nn.Parameter(th.zeros(1, 1, 1, D))
        else:
            self.tct = None

        # Temporal Embedding
        self.te = nn.Parameter(th.rand(1, T, 1, D))

        # SpecTNT blocks
        self.spectnt_blocks = nn.ModuleList([
            SpecTNTBlock(
                n_channels,
                n_frequencies,
                n_times,
                spectral_dmodel,
                spectral_nheads,
                spectral_dimff,
                temporal_dmodel,
                temporal_nheads,
                temporal_dimff,
                embed_dim,
                dropout,
                use_tct
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        """
        Input:
            x: [B, T, F, K]

        Output:
            spec_emb: [B, T, F+1, K]
            temp_emb: [B, T, 1, D]
        """
        batch_size = len(x)

        # Initialize spectral embedding - concat FCT (first raw) + add FPE
        fct = th.repeat_interleave(self.fct, batch_size, 0)  # [B, T, 1, K]
        spec_emb = th.cat([fct, x], dim=2)  # [B, T, F+1, K]
        spec_emb = spec_emb + self.fpe
        if self.tct is not None:
            spec_emb = nn.functional.pad(
                spec_emb, (0, 0, 0, 0, 1, 0))  # [B, T+1, F+1, K]

        # Initialize temporal embedding
        temp_emb = th.repeat_interleave(self.te, batch_size, 0)  # [B, T, 1, D]
        if self.tct is not None:
            tct = th.repeat_interleave(self.tct, batch_size, 0)  # [B, 1, 1, D]
            temp_emb = th.cat([tct, temp_emb], dim=1)  # [B, T+1, 1, D]

        # SpecTNT blocks inference
        for block in self.spectnt_blocks:
            spec_emb, temp_emb = block(spec_emb, temp_emb)

        return spec_emb, temp_emb


class SpecTNT(nn.Module):
    def __init__(
        self, fe_model,
        n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, n_blocks, dropout, use_tct, n_classes
    ):
        super().__init__()
        
        # TCT: Temporal Class Token
        self.use_tct = use_tct

        # Front-end model
        self.fe_model = fe_model

        # Main model
        self.main_model = SpecTNTModule(
            n_channels,
            n_frequencies,
            n_times,
            spectral_dmodel,
            spectral_nheads,
            spectral_dimff,
            temporal_dmodel,
            temporal_nheads,
            temporal_dimff,
            embed_dim,
            n_blocks,
            dropout,
            use_tct
        )
        
        # Linear layer
        self.linear_out = nn.Linear(embed_dim, n_classes)
        
    def forward(self, features):
        """
        Input:
            features: [B, K, F, T]
        
        Output:
            logits: 
                - [B, n_classes] if use_tct
                - [B, T, n_classes] otherwise
        """
        # Add channel dimension if None
        if len(features.size()) == 3:
            features = features.unsqueeze(1)
        # Front-end model
        fe_out = self.fe_model(features)            # [B, ^K, ^F, ^T]
        fe_out = fe_out.permute(0, 3, 2, 1)         # [B, T, F, K]
        # Main model
        _, temp_emb = self.main_model(fe_out)       # [B, T, 1, D]
        # Linear layer
        if self.use_tct:
            return self.linear_out(temp_emb[:, 0, 0, :])   # [B, n_classes]
        else:
            return self.linear_out(temp_emb[:, :, 0, :])   # [B, T, n_classes]


# In[ ]:


##DATASETS BLOCK##

class DummyBeatDataset(tud.Dataset):

    def __init__(self, sample_rate, input_length, hop_length, time_shrinking, mode):
        self.sample_rate = sample_rate
        self.input_length = input_length

        self.target_fps = sample_rate / (hop_length * time_shrinking)
        self.target_nframes = int(input_length * self.target_fps)

        assert mode in ["train", "validation", "test"]
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return 80
        elif self.mode == "validation":
            return 10
        elif self.mode == "test":
            return 10

    def __getitem__(self, i):
        if self.mode == "train":
            return {
                'audio': th.zeros(self.input_length * self.sample_rate),
                'targets': th.zeros(self.target_nframes, 3)
            }
        elif self.mode in ["validation", "test"]:
            return {
                'audio': th.zeros(10 * self.input_length * self.sample_rate),
                'targets': th.zeros(10 * self.target_nframes, 3),
                'beats': th.arange(0, 50, 0.5),
                'downbeats': th.arange(0, 50, 2.)
            }


class DummyBeatDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, pin_memory, sample_rate, input_length, hop_length, time_shrinking):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate
        self.input_length = input_length
        self.hop_length = hop_length
        self.time_shrinking = time_shrinking
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = True


    def setup(self, stage):
        self.train_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "train"
        )
        self.val_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "validation"
        )
        self.test_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "test"
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


# In[ ]:



class BaseModel(pl.LightningModule):
    def __init__(self, feature_extractor, net, optimizer, lr_scheduler, criterion, datamodule, activation_fn):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.datamodule = datamodule
        
        if activation_fn == "softmax":
            self.activation = nn.Softmax(dim=2)
        elif activation_fn == "sigmoid":
            self.activation = nn.Sigmoid()

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_loss"}

    @staticmethod
    def _classname(obj, lower=True):
        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = obj.__class__.__name__
        return name.lower() if lower else name


# In[ ]:



class BeatEstimator(BaseModel):
    def __init__(self, feature_extractor, net, optimizer, lr_scheduler, criterion, datamodule, activation_fn):
        super().__init__(
            feature_extractor,
            net,
            optimizer,
            lr_scheduler,
            criterion,
            datamodule,
            activation_fn
        )
        
        self.target_fps = datamodule.sample_rate /             (datamodule.hop_length * datamodule.time_shrinking)

    def training_step(self, batch, batch_idx):
        losses = {}
        x, y = batch['audio'], batch['targets']
        features = self.feature_extractor(x)
        logits = self.net(features)
        losses['train_loss'] = self.criterion(
            logits.flatten(end_dim=1), y.flatten(end_dim=1))
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['train_loss']

    def validation_step(self, batch, batch_idx):
        losses = {}
        audio, targets, ref_beats, ref_downbeats = (
            batch['audio'][0], 
            batch['targets'][0].cpu(), 
            batch['beats'][0].cpu(), 
            batch['downbeats'][0].cpu()
        )
        input_length, sample_rate, batch_size = (
            self.datamodule.input_length,
            self.datamodule.sample_rate,
            self.datamodule.batch_size
        )
        audio_chunks = th.cat([el.unsqueeze(0) for el in audio.split(
            split_size=int(input_length*sample_rate))[:-1]], dim=0)
        # Inference loop
        logits_list, probs_list = th.tensor([]), th.tensor([])
        for batch_audio in audio_chunks.split(batch_size):
            with th.no_grad():
                features = self.feature_extractor(batch_audio)
                logits = self.net(features)
                probs = self.activation(logits)
                logits_list = th.cat(
                    [logits_list, logits.flatten(end_dim=1).cpu()], dim=0)
                probs_list = th.cat(
                    [probs_list, probs.flatten(end_dim=1).cpu()], dim=0)
        # Postprocessing
        beats_data = probs_list.argmax(dim=1)
        est_beats = th.where(beats_data == 0)[0] / self.target_fps
        est_downbeats = th.where(beats_data == 1)[0] / self.target_fps
        # Eval
        losses['val_loss'] = self.criterion(
            logits_list, targets[:len(logits_list)])
        losses['beats_f_measure'] = mir_eval.beat.f_measure(
            ref_beats, est_beats)
        losses['downbeats_f_measure'] = mir_eval.beat.f_measure(
            ref_downbeats, est_downbeats)
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['val_loss']


# In[ ]:


trainer = pl.Trainer(precision=32, accumulate_grad_batches= 16, check_val_every_n_epoch= 5, max_steps= 1000000)

feature_extractor = HarmonicSTFT(sample_rate=16000, n_fft=512, n_harmonic=6, semitone_scale=2, learn_bw = 'only_Q')
fe_model = ResFrontEnd(in_channels=6, out_channels=256, freq_pooling=[2,2,2], time_pooling=[2,2,1])
net = SpecTNT(fe_model = fe_model, n_channels=256, n_frequencies=16, n_times=78, embed_dim=128, spectral_dmodel=64, spectral_nheads=4, spectral_dimff=64,
                           temporal_dmodel=256, temporal_nheads=8, temporal_dimff=256, n_blocks=5, dropout=0.15, use_tct=False, n_classes=3)
optimizer = th.optim.AdamW(params=net.parameters())
criterion = th.nn.CrossEntropyLoss()
datamodule = DummyBeatDataModule(batch_size=2, n_workers=4, pin_memory=False, sample_rate=16000, input_length=5,
                                                    hop_length=256, time_shrinking=4)
model = BeatEstimator(feature_extractor=feature_extractor, net=net, optimizer=optimizer, 
                                       lr_scheduler=None, criterion=criterion, datamodule=datamodule, activation_fn= 'softmax')

logger = pl.loggers.tensorboard.TensorBoardLogger(name = "", save_dir= "Logger")


# In[ ]:


trainer.fit(model=model, datamodule=datamodule)


# In[ ]:





# In[ ]:


##TRAIN BLOCK##




