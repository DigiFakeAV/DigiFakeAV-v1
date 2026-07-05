"""
Convert the original MP4 to the (frames/ + audio.wav) structure and generate a split list.
Requires: ffmpeg to be installed
"""
import os, sys, glob, json, random, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

SRC_ROOT = "../DigiFakeAV"
DST_ROOT = "../DigiFakeAV_processed"

REAL_DIRS = ["real_videos"]
FAKE_DIRS = [
    "FVFA_echomimic_25001_29000",
    "FVFA_hallo_29001_35000",
    "FVFA_sonic_35001_50000",
    "FVRA_echomimic_00001_05000",
    "FVRA_hallo1_05001_10000",
    "FVRA_sonic_10001_20000",
    "FVRA_sonic_20076_25000",
]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_one(args):
    src_video, out_dir, fps, size, sr = args
    frames_dir = os.path.join(out_dir, "frames")
    audio_path = os.path.join(out_dir, "audio.wav")
    done_flag = os.path.join(out_dir, ".done")
    if os.path.exists(done_flag):
        return True, src_video
    os.makedirs(frames_dir, exist_ok=True)

    r1 = run([
        "ffmpeg", "-y", "-i", src_video,
        "-vf", f"fps={fps},scale={size}:{size}",
        "-q:v", "2",
        os.path.join(frames_dir, "%05d.jpg")
    ])

    r2 = run([
        "ffmpeg", "-y", "-i", src_video,
        "-vn", "-ac", "1", "-ar", str(sr),
        audio_path
    ])
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:

        run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"anullsrc=r={sr}:cl=mono", "-t", "2", audio_path
        ])
    frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if len(frames) < 4:
        return False, src_video
    open(done_flag, "w").close()
    return True, src_video


def collect_videos():
    tasks = []          
    # real
    for d in REAL_DIRS:
        for f in sorted(glob.glob(os.path.join(SRC_ROOT, d, "**", "*"), recursive=True)):
            if f.lower().endswith(VIDEO_EXTS):
                name = f"{d}__{os.path.splitext(os.path.basename(f))[0]}"
                dst = os.path.join(DST_ROOT, "real", name)
                tasks.append((f, dst, 0))
    # fake
    for d in FAKE_DIRS:
        for f in sorted(glob.glob(os.path.join(SRC_ROOT, d, "**", "*"), recursive=True)):
            if f.lower().endswith(VIDEO_EXTS):
                name = f"{d}__{os.path.splitext(os.path.basename(f))[0]}"
                dst = os.path.join(DST_ROOT, "fake", name)
                tasks.append((f, dst, 1))
    return tasks


def main(args):
    os.makedirs(DST_ROOT, exist_ok=True)
    tasks = collect_videos()
    print(
          f"(real={sum(1 for _,_,l in tasks if l==0)}, "
          f"fake={sum(1 for _,_,l in tasks if l==1)})")

    job_args = [(src, dst, args.fps, args.size, args.sr) for src, dst, _ in tasks]
    ok_map = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_one, ja): ja for ja in job_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ffmpeg"):
            ok, src = fut.result()
            ok_map[src] = ok


    good = [(src, dst, label) for (src, dst, label) in tasks if ok_map.get(src, False)]
    print(f"Processed successfully {len(good)} / {len(tasks)}")


    random.seed(42)
    by_cls = {0: [], 1: []}
    for src, dst, label in good:
        by_cls[label].append(dst)
    for k in by_cls: random.shuffle(by_cls[k])

    splits = {"train": [], "val": [], "test": []}
    for label, items in by_cls.items():
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        for p in items[:n_train]:              splits["train"].append((p, label))
        for p in items[n_train:n_train+n_val]: splits["val"].append((p, label))
        for p in items[n_train+n_val:]:        splits["test"].append((p, label))


    split_dir = os.path.join(DST_ROOT, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for name, lst in splits.items():
        random.shuffle(lst)
        with open(os.path.join(split_dir, f"{name}.txt"), "w") as f:
            for p, l in lst:
                f.write(f"{p}\t{l}\n")
        print(f"{name}: {len(lst)}  (real={sum(1 for _,l in lst if l==0)}, "
              f"fake={sum(1 for _,l in lst if l==1)})")

    print(f"\nDone. Root directory after processing: {DST_ROOT}")
    print(f"   The splits file is located at: {split_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--num_workers", type=int, default=16)
    main(p.parse_args())