import os

def _load_txt(txt_path, root):
    paths, labels = [], []
    if not os.path.isfile(txt_path):
        return paths, labels

    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2: continue
            p, lbl = parts[0], int(parts[1])


            if os.path.isdir(p):
                paths.append(p); labels.append(lbl); continue

            norm = p.replace('\\', '/')
            for key in ['/fake/', '/real/']:
                if key in norm:
                    rel = norm.split(key, 1)[1]
                    cls = key.strip('/')
                    cand = os.path.join(root, cls, rel)
                    if os.path.isdir(cand):
                        paths.append(cand); labels.append(lbl)
                        break
            else:
                cand = os.path.join(root, p)
                if os.path.isdir(cand):
                    paths.append(cand); labels.append(lbl)

    return paths, labels


def read_split_data(root):
    split_dir = os.path.join(root, "splits")
    tr_p, tr_l = _load_txt(os.path.join(split_dir, "train.txt"), root)
    val_p, val_l = _load_txt(os.path.join(split_dir, "val.txt"), root)
    print(f"[Data] train={len(tr_p)} (real={tr_l.count(0)}, fake={tr_l.count(1)}) | "
          f"val={len(val_p)} (real={val_l.count(0)}, fake={val_l.count(1)})")
    return tr_p, tr_l, val_p, val_l


def read_test_data(root):
    p, l = _load_txt(os.path.join(root, "splits", "test.txt"), root)
    print(f"[Test] {len(p)} videos (real={l.count(0)}, fake={l.count(1)})")
    return p, l