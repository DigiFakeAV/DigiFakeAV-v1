import os, argparse, json, time

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='0')
_pa, _ = _pre.parse_known_args()
if _pa.device.lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = _pa.device
# ------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, roc_curve, confusion_matrix)

from utils.preprocess import read_test_data
from utils.YouxiangDataSet import MyDataset
from models.DigiShield_main import DigiShield


def _unpack(o):
    if isinstance(o, (list, tuple)):
        return o[-1]
    return o


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total, correct, loss_sum = 0, 0, 0.0
    all_scores, all_labels, all_paths = [], [], []
    pbar = tqdm(loader, desc='Test', ncols=110)
    for batch in pbar:
        v = batch['visual'].to(device, non_blocking=True)
        a = batch['audio'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        logits = _unpack(model(v, a))
        loss = ce(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(1)
        loss_sum += loss.item()
        correct  += (pred == y).sum().item()
        total    += y.size(0)
        all_scores.extend(prob.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
        if 'path' in batch:
            all_paths.extend(batch['path'])
        pbar.set_postfix(acc=f"{correct/total:.4f}")
    return (loss_sum / max(total, 1), correct / max(total, 1),
            np.asarray(all_labels), np.asarray(all_scores), all_paths)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device != 'cpu'
                          else 'cpu')
    print(f"[Device] {device}")

    # ---- data ----
    te_paths, te_labels = read_test_data(args.data_path)
    print(f"[Data] test={len(te_paths)} "
          f"(real={te_labels.count(0)}, fake={te_labels.count(1)})")
    te_set = MyDataset(te_paths, te_labels,
                       num_frames=args.num_frames, train=False)
    te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=min(os.cpu_count(), 8), pin_memory=True,
                           collate_fn=MyDataset.collate_fn)

    # ---- model ----
    model = DigiShield(d=1024, n_fusion=args.n_fusion).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt.get('state_dict', ckpt)

    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Ckpt] loaded {args.ckpt} | epoch={ckpt.get('epoch','?')} "
          f"val_AUC={ckpt.get('auc','?')} | missing={len(missing)} unexpected={len(unexpected)}")

    # ---- run ----
    loss, acc, y, s, paths = run(model, te_loader, device)
    auc = roc_auc_score(y, s)
    f1  = f1_score(y, (s >= 0.5).astype(int))
    eer, eer_thr = compute_eer(y, s)
    cm  = confusion_matrix(y, (s >= 0.5).astype(int)).tolist()

    print("\n================ TEST RESULT ================")
    print(f"samples   : {len(y)}")
    print(f"loss      : {loss:.4f}")
    print(f"accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"EER       : {eer:.4f}   (thr={eer_thr:.3f})")
    print(f"confusion : {cm}   [rows=GT(0,1), cols=Pred(0,1)]")
    print("=============================================")

    out = {
        'ckpt': args.ckpt,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': int(len(y)),
        'loss': float(loss), 'accuracy': float(acc),
        'auc': float(auc), 'f1': float(f1),
        'eer': float(eer), 'eer_threshold': float(eer_thr),
        'confusion_matrix': cm,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # per-sample 分數
    with open(os.path.join(args.out_dir, 'test_scores.tsv'), 'w') as f:
        f.write("path\tlabel\tscore_fake\tpred\n")
        for i in range(len(y)):
            p = paths[i] if paths else te_paths[i]
            f.write(f"{p}\t{int(y[i])}\t{float(s[i]):.6f}\t{int(s[i]>=0.5)}\n")

    print(f"[Save] {args.out_dir}/test_metrics.json")
    print(f"[Save] {args.out_dir}/test_scores.tsv")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', dest='data_path',
                   default='../DigiFakeAV_processed')
    p.add_argument('--ckpt',
                   default='../DigiShield/weights/DigiShield_best.pth')
    p.add_argument('--out_dir',
                   default='../DigiShield/weights/test_out')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_frames', type=int, default=16)
    p.add_argument('--n_fusion',   type=int, default=2)
    p.add_argument('--device', default='0')
    main(p.parse_args())