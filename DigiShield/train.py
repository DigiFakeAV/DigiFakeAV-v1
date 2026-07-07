import os, math, argparse, random, json, time

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='0,1,2,3,4,5')
_pa, _ = _pre.parse_known_args()
if _pa.device.lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = _pa.device

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from utils.preprocess import read_split_data, read_test_data
from utils.YouxiangDataSet import ClipDataset
from utils.train_epoch import train_one_epoch, evaluate
from models.DigiShield_main import DigiShield


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        labels = np.asarray(labels)
        self.real_idx = np.where(labels == 0)[0].tolist()
        self.fake_idx = np.where(labels == 1)[0].tolist()
        self.n_per_cls = batch_size // 2
        self.n_batches = min(len(self.real_idx), len(self.fake_idx)) // self.n_per_cls
        if not drop_last:
            self.n_batches = max(len(self.real_idx), len(self.fake_idx)) // self.n_per_cls

    def __iter__(self):
        random.shuffle(self.real_idx)
        random.shuffle(self.fake_idx)
        for i in range(self.n_batches):
            real_batch = self.real_idx[i * self.n_per_cls : (i + 1) * self.n_per_cls]
            fake_batch = self.fake_idx[i * self.n_per_cls : (i + 1) * self.n_per_cls]
            batch = real_batch + fake_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def main(args):
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    n_gpu = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f"[Device] {n_gpu} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
    os.makedirs(args.weights, exist_ok=True)

    tr_paths, tr_labels, val_paths, val_labels = read_split_data(args.data_path)
    tr_set = ClipDataset(tr_paths, tr_labels, clip_len=args.num_frames, stride=args.num_frames, train=True)
    val_set = ClipDataset(val_paths, val_labels, clip_len=args.num_frames, stride=args.num_frames, train=False)

    tr_sampler = BalancedBatchSampler([c[1] for c in tr_set.clips], args.batch_size, drop_last=True)
    nw = min(os.cpu_count(), 8)
    tr_loader = DataLoader(tr_set, batch_sampler=tr_sampler,
                           num_workers=nw, pin_memory=True,
                           collate_fn=ClipDataset.collate_fn, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            collate_fn=ClipDataset.collate_fn, persistent_workers=(nw > 0))

    # ---- Model ----
    model = DigiShield(d=1024, n_fusion=args.n_fusion)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ---- Optim ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warmup = args.warmup
    def lr_fn(step):
        if step < warmup: return (step + 1) / warmup
        prog = (step - warmup) / max(1, args.epochs - warmup)
        return max(args.lrf, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    tb = SummaryWriter(log_dir=os.path.join(args.weights, 'logs'))
    best_auc, best_ep = 0.0, -1
    ckpt_best = os.path.join(args.weights, f"{args.model_name}_best.pth")

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, tr_loader, device, epoch, scaler)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, epoch)
        scheduler.step()
        print(f"[Epoch {epoch}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_AUC={val_auc:.4f}")
        for k, v in {"tr_loss": tr_loss, "tr_acc": tr_acc, "val_loss": val_loss,
                     "val_acc": val_acc, "val_auc": val_auc, "lr": optimizer.param_groups[0]['lr']}.items():
            tb.add_scalar(k, v, epoch)

        if val_auc > best_auc:
            best_auc, best_ep = val_auc, epoch
            sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'state_dict': sd, 'auc': best_auc, 'epoch': epoch}, ckpt_best)
            print(f"💾 Best → {ckpt_best} (AUC={best_auc:.4f})")

    print(f"\n✅ Training Complete | Best val AUC={best_auc:.4f} @ epoch {best_ep}")

    # ---- Test ----
    try:
        te_paths, te_labels = read_test_data(args.data_path)
        te_set = ClipDataset(te_paths, te_labels, clip_len=args.num_frames, stride=args.num_frames, train=False)
        te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=nw, pin_memory=True, collate_fn=ClipDataset.collate_fn)
        ckpt = torch.load(ckpt_best, map_location=device)
        sd = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd, strict=False)
        te_loss, te_acc, te_auc = evaluate(model, te_loader, device, -1)
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} AUC={te_auc:.4f}")
        with open(os.path.join(args.weights, 'test_result.json'), 'w') as f:
            json.dump({'best_val_auc': float(best_auc), 'test_auc': float(te_auc),
                       'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    except Exception as e:
        print(f"[Test] Skip: {e}")
    tb.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=32) 
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--num_frames', type=int, default=30)
    p.add_argument('--n_fusion', type=int, default=2)
    p.add_argument('--data-path', dest='data_path',
                   default='../DigiFakeAV_processed')
    p.add_argument('--model-name', default='DigiShield')
    p.add_argument('--weights', default='../DigiShield/weights')
    p.add_argument('--device', default='0,1,2,3,4,5')
    main(p.parse_args())