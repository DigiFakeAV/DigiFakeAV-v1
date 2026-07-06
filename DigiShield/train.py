
import os, math, argparse, random, json, time

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='0,7')
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device.lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.device

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.preprocess import read_split_data, read_test_data
from utils.YouxiangDataSet import MyDataset
from utils.train_epoch import train_one_epoch, evaluate
from models.DigiShield_main import DigiShield


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_balanced_sampler(labels):
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.size > 0, "Training label is empty"
    cnt = np.bincount(labels)
    w = 1.0 / np.maximum(cnt, 1)
    sw = w[labels]
    return WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                 num_samples=len(sw), replacement=True)


def main(args):
    set_seed(42)
    if args.device.lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu'); n_gpu = 0
    else:
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0')
        print(f"[Device] {n_gpu} GPU(s) visible: "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
    os.makedirs(args.weights, exist_ok=True)

    # ---------- data ----------
    train_paths, train_labels, val_paths, val_labels = read_split_data(args.data_path)
    train_set = MyDataset(train_paths, train_labels, num_frames=args.num_frames, train=True)
    val_set   = MyDataset(val_paths,   val_labels,   num_frames=args.num_frames, train=False)

    sampler = build_balanced_sampler(train_labels)
    nw = min(os.cpu_count(), 8)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              collate_fn=MyDataset.collate_fn,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            collate_fn=MyDataset.collate_fn,
                            persistent_workers=(nw > 0))

    # ---------- model ----------
    model = DigiShield(d=1024, n_fusion=args.n_fusion)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ---------- optim ----------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0.05)
    warmup = args.warmup
    def lr_lambda(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, args.epochs - warmup)
        return max(args.lrf, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    tb = SummaryWriter(log_dir=os.path.join(args.weights, 'logs'))
    best_auc, best_epoch = 0.0, -1
    ckpt_best = os.path.join(args.weights, f"{args.model_name}_best.pth")
    ckpt_last = os.path.join(args.weights, f"{args.model_name}_last.pth")

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader,
                                          device, epoch, scaler)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, epoch)
        scheduler.step()
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_AUC={val_auc:.4f}")

        for k, v in dict(tr_loss=tr_loss, tr_acc=tr_acc, val_loss=val_loss,
                         val_acc=val_acc, val_auc=val_auc,
                         lr=optimizer.param_groups[0]['lr']).items():
            tb.add_scalar(k, v, epoch)

        sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        # last
        torch.save({'state_dict': sd, 'auc': val_auc, 'epoch': epoch}, ckpt_last)
        # best
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({'state_dict': sd, 'auc': best_auc, 'epoch': epoch}, ckpt_best)
            print(f" Save for later → {ckpt_best}  (val_AUC={best_auc:.4f})")

    print(f"\n===== Best val AUC={best_auc:.4f} @ epoch {best_epoch} =====")

    try:
        te_paths, te_labels = read_test_data(args.data_path)
        te_set = MyDataset(te_paths, te_labels, num_frames=args.num_frames, train=False)
        te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=nw, pin_memory=True,
                               collate_fn=MyDataset.collate_fn)
        ckpt = torch.load(ckpt_best, map_location=device)
        sd = {k.replace('module.', '', 1): v for k, v in ckpt['state_dict'].items()}
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd, strict=False)
        te_loss, te_acc, te_auc = evaluate(model, te_loader, device, epoch=-1)
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} AUC={te_auc:.4f}")
        with open(os.path.join(args.weights, 'test_result.json'), 'w') as f:
            json.dump({'best_val_auc': float(best_auc), 'best_epoch': int(best_epoch),
                       'test_loss': float(te_loss), 'test_acc': float(te_acc),
                       'test_auc': float(te_auc),
                       'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    except Exception as e:
        print(f"[Test] Skip: {e}")

    tb.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--num_frames', type=int, default=30)
    p.add_argument('--n_fusion', type=int, default=2)
    p.add_argument('--data-path', dest='data_path', type=str,
                   default='../DigiFakeAV_processed')
    p.add_argument('--model-name', default='DigiShield')
    p.add_argument('--weights', type=str, default='../DigiShield/weights')
    p.add_argument('--device', default='0,7')
    main(p.parse_args())