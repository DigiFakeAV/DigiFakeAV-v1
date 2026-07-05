import os


def _read_split(path):
    paths, labels = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, l = line.split("\t")
            paths.append(p)
            labels.append(int(l))
    return paths, labels


def read_split_data(data_root):
    split_dir = os.path.join(data_root, "splits")
    train_p, train_l = _read_split(os.path.join(split_dir, "train.txt"))
    val_p,   val_l   = _read_split(os.path.join(split_dir, "val.txt"))
    print(f"[Data] train={len(train_p)} (real={train_l.count(0)}, fake={train_l.count(1)}) | "
          f"val={len(val_p)} (real={val_l.count(0)}, fake={val_l.count(1)})")
    assert len(train_p) > 0 and len(val_p) > 0, f"“split” is empty. Please check {split_dir}"
    return train_p, train_l, val_p, val_l


def read_test_data(test_root_or_split):
    if os.path.isfile(test_root_or_split):
        return _read_split(test_root_or_split)
    fp = os.path.join(test_root_or_split, "splits", "test.txt")
    if os.path.isfile(fp):
        return _read_split(fp)
    raise FileNotFoundError(f"Cannot find “test split”: {test_root_or_split}")