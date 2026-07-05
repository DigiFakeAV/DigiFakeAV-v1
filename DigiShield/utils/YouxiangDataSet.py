import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

try:
    import torchaudio
    _HAS_TA = True
except Exception:
    _HAS_TA = False

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _list_frames(sample_dir):
    for sub in ["frames", "frame", "imgs", "images"]:
        d = os.path.join(sample_dir, sub)
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "*.jpg")) +
                           glob.glob(os.path.join(d, "*.png")))
            if files:
                return files

    files = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")) +
                   glob.glob(os.path.join(sample_dir, "*.png")))
    return files


def _find_audio(sample_dir):
    for name in ["audio.wav", "audio.flac", "audio.mp3"]:
        p = os.path.join(sample_dir, name)
        if os.path.isfile(p):
            return p
    cands = sorted(glob.glob(os.path.join(sample_dir, "*.wav")) +
                   glob.glob(os.path.join(sample_dir, "*.flac")) +
                   glob.glob(os.path.join(sample_dir, "*.mp3")))
    return cands[0] if cands else None


class MyDataset(Dataset):
    """
        sample_dir/
            frames/00001.jpg, 00002.jpg, ...  
            audio.wav
    """
    def __init__(self, sample_dirs, labels, num_frames=20,
                 img_size=224, sr=16000, mel_bins=128,
                 audio_seconds=2.0, train=True):
        assert len(sample_dirs) == len(labels)
        self.dirs = list(sample_dirs)
        self.labels = list(labels)
        self.num_frames = num_frames
        self.img_size = img_size
        self.sr = sr
        self.mel_bins = mel_bins
        self.audio_len = int(sr * audio_seconds)
        self.train = train

        if train:
            self.frame_tf = T.Compose([
                T.Resize((img_size + 16, img_size + 16)),
                T.RandomCrop(img_size),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.Normalize(IMG_MEAN, IMG_STD),
            ])
        else:
            self.frame_tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(IMG_MEAN, IMG_STD),
            ])

        if _HAS_TA:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=1024, hop_length=256,
                n_mels=mel_bins, power=2.0)
            self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.dirs)

    def _load_frames(self, sample_dir):
        files = _list_frames(sample_dir)
        if len(files) == 0:

            return torch.zeros(self.num_frames, 3, self.img_size, self.img_size)

        n = len(files)
        if n >= self.num_frames:
            if self.train:
                idxs = np.linspace(0, n, self.num_frames + 1).astype(int)
                pick = [random.randint(idxs[i], max(idxs[i], idxs[i + 1] - 1))
                        for i in range(self.num_frames)]
            else:
                pick = np.linspace(0, n - 1, self.num_frames).astype(int).tolist()
        else:
            pick = [i % n for i in range(self.num_frames)]

        do_flip = self.train and (random.random() < 0.5)
        imgs = []
        for i in pick:
            try:
                img = Image.open(files[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.img_size, self.img_size))
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            imgs.append(self.frame_tf(img))
        return torch.stack(imgs, dim=0)  

    def _load_waveform(self, wav_path):
        try:
            if _HAS_TA:
                wav, sr = torchaudio.load(wav_path)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                wav = wav.mean(0)
                return wav
            elif _HAS_SF:
                w, sr = sf.read(wav_path)
                if w.ndim > 1:
                    w = w.mean(-1)
                w = torch.from_numpy(np.asarray(w)).float()
                if sr != self.sr:

                    n_new = int(len(w) * self.sr / sr)
                    w = torch.nn.functional.interpolate(
                        w.view(1, 1, -1), size=n_new, mode="linear", align_corners=False
                    ).view(-1)
                return w
        except Exception:
            pass
        return torch.zeros(self.audio_len)

    def _load_audio(self, sample_dir):
        wav_path = _find_audio(sample_dir)
        wav = self._load_waveform(wav_path) if wav_path else torch.zeros(self.audio_len)
        if wav.numel() == 0:
            wav = torch.zeros(self.audio_len)

        if wav.numel() < self.audio_len:
            wav = torch.nn.functional.pad(wav, (0, self.audio_len - wav.numel()))
        else:
            if self.train:
                start = random.randint(0, wav.numel() - self.audio_len)
            else:
                start = (wav.numel() - self.audio_len) // 2
            wav = wav[start:start + self.audio_len]

        if _HAS_TA:
            mel = self.db(self.mel(wav.unsqueeze(0))).squeeze(0)  
        else:
            spec = torch.stft(wav, n_fft=1024, hop_length=256,
                              return_complex=True).abs().pow(2)
            mel = torch.log(spec + 1e-6)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0).unsqueeze(0) 
        mel = torch.nn.functional.interpolate(
            mel, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False)
        mel = mel.squeeze(0).repeat(3, 1, 1)  
        return mel

    def __getitem__(self, idx):
        sd = self.dirs[idx]
        label = int(self.labels[idx])
        try:
            visual = self._load_frames(sd)
            audio = self._load_audio(sd)
        except Exception as e:
            print(f"[WARN] load fail: {sd} | {e}")
            visual = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
            audio = torch.zeros(3, self.img_size, self.img_size)
        return {
            "visual": visual,
            "audio": audio,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "visual": torch.stack([b["visual"] for b in batch], dim=0),
            "audio":  torch.stack([b["audio"]  for b in batch], dim=0),
            "label":  torch.stack([b["label"]  for b in batch], dim=0),
        }