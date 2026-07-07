import os, argparse, json, time
from collections import defaultdict

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='0')
_pa, _ = _pre.parse_known_args()
if _pa.device.lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = _pa.device

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, confusion_matrix)

from utils.preprocess import read_test_data
from utils.YouxiangDataSet import ClipDataset
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
    clip_scores, clip_labels, clip_dirs = [], [], []

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
        clip_scores.extend(prob.detach().cpu().tolist())
        clip_labels.extend(y.detach().cpu().tolist())
        clip_dirs.extend(batch['video_dir'])
        pbar.set_postfix(acc=f"{correct/total:.4f}")

    return (loss_sum / max(total, 1),
            correct / max(total, 1),
            np.asarray(clip_labels), np.asarray(clip_scores), clip_dirs)


def aggregate_video_level(clip_scores, clip_labels, clip_dirs):
    v_scores = defaultdict(list)
    v_label = {}
    for s, y, d in zip(clip_scores, clip_labels, clip_dirs):
        v_scores[d].append(float(s))
        v_label[d] = int(y)

    dirs, ys, ss = [], [], []
    for d, plist in v_scores.items():
        dirs.append(d)
        ys.append(v_label[d])
        ss.append(sum(plist) / len(plist))
    return np.asarray(ys), np.asarray(ss), dirs


def report(tag, y, s, out_dict):
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else 0.5
    pred = (s >= 0.5).astype(int)
    acc = (pred == y).mean()
    f1  = f1_score(y, pred)
    eer, eer_thr = compute_eer(y, s)
    cm  = confusion_matrix(y, pred).tolist()
    print(f"\n================ {tag} RESULT ================")
    print(f"samples   : {len(y)}")
    print(f"accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"EER       : {eer:.4f}   (thr={eer_thr:.3f})")
    print(f"confusion : {cm}   [rows=GT(0,1), cols=Pred(0,1)]")
    print("=" * (len(tag) + 30))
    out_dict[tag.lower()] = {
        'accuracy': float(acc), 'auc': float(auc), 'f1': float(f1),
        'eer': float(eer), 'eer_threshold': float(eer_thr),
        'confusion_matrix': cm,
    }
    return auc


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device != 'cpu'
                          else 'cpu')
    print(f"[Device] {device}")

    te_paths, te_labels = read_test_data(args.data_path)
    stride = max(1, args.num_frames // 2) if args.overlap else args.num_frames
    te_set = ClipDataset(te_paths, te_labels,
                         clip_len=args.num_frames, stride=stride, train=False)
    te_loader = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=min(os.cpu_count(), 8), pin_memory=True,
                           collate_fn=ClipDataset.collate_fn)

    # ---- model ----
    model = DigiShield(d=1024, n_fusion=args.n_fusion).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Ckpt] loaded {args.ckpt} | epoch={ckpt.get('epoch','?')} "
          f"val_AUC={ckpt.get('auc','?')} | missing={len(missing)} unexpected={len(unexpected)}")

    # ---- run inference (clip level) ----
    loss, clip_acc, y_clip, s_clip, dirs_clip = run(model, te_loader, device)
    print(f"\n[Loss] {loss:.4f}")

    out = {
        'ckpt': args.ckpt,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'clip_len': args.num_frames,
        'stride': stride,
        'loss': float(loss),
    }

    report("CLIP", y_clip, s_clip, out)

    y_vid, s_vid, dirs_vid = aggregate_video_level(s_clip, y_clip, dirs_clip)
    report("VIDEO", y_vid, s_vid, out)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'test_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, 'test_scores_clip.tsv'), 'w') as f:
        f.write("video_dir\tlabel\tscore_fake\tpred\n")
        for i in range(len(y_clip)):
            f.write(f"{dirs_clip[i]}\t{int(y_clip[i])}\t{float(s_clip[i]):.6f}\t{int(s_clip[i]>=0.5)}\n")

    with open(os.path.join(args.out_dir, 'test_scores_video.tsv'), 'w') as f:
        f.write("video_dir\tlabel\tscore_fake\tpred\n")
        for i in range(len(y_vid)):
            f.write(f"{dirs_vid[i]}\t{int(y_vid[i])}\t{float(s_vid[i]):.6f}\t{int(s_vid[i]>=0.5)}\n")

    print(f"\n[Save] {args.out_dir}/test_metrics.json")
    print(f"[Save] {args.out_dir}/test_scores_clip.tsv")
    print(f"[Save] {args.out_dir}/test_scores_video.tsv")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', dest='data_path',
                   default='../DigiFakeAV_processed')
    p.add_argument('--ckpt',
                   default='../DigiShield/weights/DigiShield_best.pth')
    p.add_argument('--out_dir',
                   default='../DigiShield/weights/test_out')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_frames', type=int, default=30)
    p.add_argument('--n_fusion',   type=int, default=2)
    p.add_argument('--overlap', action='store_true')
    p.add_argument('--device', default='0')
    main(p.parse_args())