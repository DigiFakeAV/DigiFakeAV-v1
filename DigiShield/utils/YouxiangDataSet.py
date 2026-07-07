import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

try:
    import torchaudio
    _HAS_TA = True
except:
    _HAS_TA = False

try:
    import soundfile as sf
    _HAS_SF = True
except:
    _HAS_SF = False

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def _list_frames(d):
    for sub in ["frames", "frame"]:
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            fs = sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png")))
            if fs: return fs
    return sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))


def _find_audio(d):
    for n in ["audio.wav", "audio.flac", "audio.mp3"]:
        p = os.path.join(d, n)
        if os.path.isfile(p): return p
    cands = sorted(glob.glob(os.path.join(d, "*.wav")) + glob.glob(os.path.join(d, "*.flac")))
    return cands[0] if cands else None


class ClipDataset(Dataset):

    def __init__(self, sample_dirs, labels, clip_len=30, stride=None,
                 img_size=224, sr=16000, mel_bins=128, train=True):
        self.clip_len = clip_len
        self.stride = stride if stride else clip_len
        self.img_size, self.sr, self.mel_bins = img_size, sr, mel_bins
        self.train = train

        self.clips = []
        for d, lbl in zip(sample_dirs, labels):
            frames = _list_frames(d)
            if len(frames) < clip_len: continue 
            for start in range(0, len(frames) - clip_len + 1, self.stride):
                self.clips.append((d, int(lbl), start))

        print(f"[ClipDataset] {len(sample_dirs)} videos → {len(self.clips)} clips (train={train})")

        if train:
            self.frame_tf = T.Compose([
                T.Resize((img_size + 16, img_size + 16)),
                T.RandomCrop(img_size),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomHorizontalFlip(0.5),
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
                sample_rate=sr, n_fft=1024, hop_length=256, n_mels=mel_bins, power=2.0)
            self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.clips)

    def _load_frames(self, d, start):
        files = _list_frames(d)
        end = min(start + self.clip_len, len(files))
        pick = list(range(start, end))
        if len(pick) < self.clip_len:  
            pick += [pick[-1]] * (self.clip_len - len(pick))

        imgs = []
        for i in pick:
            try:
                img = Image.open(files[i]).convert("RGB")
            except:
                img = Image.new("RGB", (self.img_size, self.img_size))
            imgs.append(self.frame_tf(img))
        return torch.stack(imgs, dim=0) 
    def _load_audio(self, d, start):
        wav_p = _find_audio(d)
        fps = 25  
        start_sec = start / fps
        dur_sec = self.clip_len / fps
        n_samples = int(dur_sec * self.sr)

        if wav_p and os.path.isfile(wav_p):
            try:
                if _HAS_TA:
                    wav, sr = torchaudio.load(wav_p)
                    if sr != self.sr:
                        wav = torchaudio.functional.resample(wav, sr, self.sr)
                    wav = wav.mean(0) if wav.dim() > 1 else wav.squeeze()
                elif _HAS_SF:
                    w, sr = sf.read(wav_p)
                    if w.ndim > 1: w = w.mean(-1)
                    wav = torch.from_numpy(w).float()
                    if sr != self.sr:
                        n_new = int(len(wav) * self.sr / sr)
                        wav = torch.nn.functional.interpolate(
                            wav.view(1,1,-1), size=n_new, mode='linear', align_corners=False).view(-1)
                else:
                    wav = torch.zeros(n_samples)


                offset = int(start_sec * self.sr)
                if offset + n_samples <= wav.numel():
                    wav = wav[offset : offset + n_samples]
                else:
                    wav = torch.nn.functional.pad(wav[offset:], (0, n_samples - (wav.numel() - offset)))
            except:
                wav = torch.zeros(n_samples)
        else:
            wav = torch.zeros(n_samples)

        # Mel
        if _HAS_TA:
            mel = self.db(self.mel(wav.unsqueeze(0))).squeeze(0)
        else:
            spec = torch.stft(wav, n_fft=1024, hop_length=256, return_complex=True).abs().pow(2)
            mel = torch.log(spec + 1e-6)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0).unsqueeze(0)  
        mel = torch.nn.functional.interpolate(mel, size=(self.img_size, self.img_size),
                                              mode='bilinear', align_corners=False)
        return mel.squeeze(0).repeat(3, 1, 1)

    def __getitem__(self, idx):
        d, lbl, start = self.clips[idx]
        try:
            v = self._load_frames(d, start)
            a = self._load_audio(d, start)
        except Exception as e:
            print(f"[WARN] {d} | {e}")
            v = torch.zeros(self.clip_len, 3, self.img_size, self.img_size)
            a = torch.zeros(3, self.img_size, self.img_size)
        return {
            "visual": v,
            "audio": a,
            "label": torch.tensor(lbl, dtype=torch.long),
            "video_dir": d,  
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "visual": torch.stack([b["visual"] for b in batch], dim=0),
            "audio":  torch.stack([b["audio"]  for b in batch], dim=0),
            "label":  torch.stack([b["label"]  for b in batch], dim=0),
            "video_dir": [b["video_dir"] for b in batch],
        }