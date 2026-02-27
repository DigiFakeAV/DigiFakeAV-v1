import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class MyDataset(Dataset):
    def __init__(self, 
                 video_paths: list,
                 labels: list,
                 num_frames: int = 20):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        
        self.transform = T.Compose([
            T.Resize((224, 224)),  
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def _load_frames(self, path: str) -> torch.Tensor:
        frame_files = sorted(
            [f for f in os.listdir(path) if f.endswith(('.jpg','.png')) and not f.startswith('mel_spec')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )[:self.num_frames]
        return torch.stack([self.transform(Image.open(os.path.join(path, f)).convert('RGB')) 
                          for f in frame_files])

    def _load_spectrogram(self, path: str) -> torch.Tensor:
        spec_path = os.path.join(path, 'mel_spec.jpg')
        img = Image.open(spec_path).convert('RGB')
        return self.transform(img)  

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        visual = self._load_frames(video_path)
        audio = self._load_spectrogram(video_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
            

        return {
            'visual': visual,
            'audio': audio,
            'label': label
            }

    @staticmethod
    def collate_fn(batch):
        return {
            'visual': torch.stack([x['visual'] for x in batch]),  # [B, T, C, H, W]
            'audio': torch.stack([x['audio'] for x in batch]),    # [B, C, H, W]
            'label': torch.stack([x['label'] for x in batch])     # [B]
        }
