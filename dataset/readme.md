## Dataset Description

This dataset contains synthetic and real video clips as part of the **DigiFakeAV** project. Each video sample is composed of:

- **Video frames:** Stored as raw numpy arrays (`.frames.npy`), together with metadata files (`.height.txt`, `.width.txt`, `.num_frames.txt`).  
- **Audio:** Provided separately in uncompressed `.wav` format (`.audio.wav`).  
- (Optionally, labels or metadata as required.)

**Note:**  
For storage and processing efficiency, video and audio are stored as separate files. The original mp4 videos can be accurately reconstructed by combining frames and audio using the provided scripts. Currently, only the video files are uploaded; full data packages including audio are being uploaded gradually.

### Reconstructing videos

To reconstruct standard `.mp4` videos from this dataset (combining frames and audio), please use the provided Python example below:

```python
import numpy as np
import cv2
import subprocess
import os

def frames_to_temp_mp4(prefix, fps=25):
    h = int(open(prefix + ".height.txt").read())
    w = int(open(prefix + ".width.txt").read())
    n = int(open(prefix + ".num_frames.txt").read())
    frames = np.fromfile(prefix + ".frames.npy", dtype=np.uint8).reshape(n, h, w, 3)
    
    temp_mp4 = prefix + "_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_mp4

def mux_video_audio(video_path, audio_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)

def reconstruct_mp4(prefix, fps=25):
    temp_mp4 = frames_to_temp_mp4(prefix, fps=fps)
    audio_path = prefix + ".audio.wav"
    out_mp4 = prefix + ".mp4"
    mux_video_audio(temp_mp4, audio_path, out_mp4)
    os.remove(temp_mp4)
    print(f"✅ Finished: {out_mp4}")

# Example (replace with your actual prefix):
reconstruct_mp4("DigiFakeAV_real_1_500/real_videos_1")

# For batch processing:
import glob
for prefix in glob.glob("DigiFakeAV_real_1_500/real_videos_*[!.wav]"):
    if os.path.exists(prefix + ".audio.wav"):
        reconstruct_mp4(prefix)
```

**Requirements:**  
- Python packages: `numpy`, `opencv-python`
- [FFmpeg](https://ffmpeg.org/) installed and accessible from your system PATH

---
