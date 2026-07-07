import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.losses import SymInfoNCE

criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_con = SymInfoNCE(temperature=0.07, margin=1.0, use_margin=True, lambda_margin=0.3)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, grad_clip=1.0):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(data_loader, desc=f'Train {epoch}')
    for batch in pbar:
        video = batch['visual'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                v_p, a_p, logits = model(video, audio)
                l_ce = criterion_cls(logits, y)
                l_con = criterion_con(v_p, a_p, y)
                loss = l_ce + l_con
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            v_p, a_p, logits = model(video, audio)
            l_ce = criterion_cls(logits, y)
            l_con = criterion_con(v_p, a_p, y)
            loss = l_ce + l_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tot_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%",
                         ce=f"{l_ce.item():.3f}", con=f"{l_con.item():.3f}")
    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0

    clip_probs, clip_labels, clip_dirs = [], [], []
    pbar = tqdm(data_loader, desc=f'Val {epoch}')
    for batch in pbar:
        video = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        y = batch['label'].to(device)
        _, _, logits = model(video, audio)
        loss = criterion_cls(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        tot_loss += loss.item() * y.size(0)
        clip_probs.extend(prob.cpu().tolist())
        clip_labels.extend(y.cpu().tolist())
        clip_dirs.extend(batch['video_dir'])
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")


    from collections import defaultdict
    video_scores = defaultdict(list)
    video_labels = {}
    for p, lbl, d in zip(clip_probs, clip_labels, clip_dirs):
        video_scores[d].append(p)
        video_labels[d] = lbl

    y_vid = []
    s_vid = []
    for d in video_scores:
        y_vid.append(video_labels[d])
        s_vid.append(sum(video_scores[d]) / len(video_scores[d]))

    try:
        auc = roc_auc_score(y_vid, s_vid)
    except:
        auc = 0.5
    return tot_loss / total, correct / total, auc