<div align="center">
<img style="width: 60%;" src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/figure01.png">

<h2 align="center">DigiFakeAV: Beyond Face Swapping — A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection</h2>

<a href="https://arxiv.org/abs/2001.03024" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-DigiFakeAV-red?style=flat&logo=arXiv" alt="Paper PDF" height="25">
</a>
<a href="https://hubeiwuhanliu.github.io/DigiFakeAV.github.io/" target="_blank">
  <img alt="Website" src="https://img.shields.io/badge/🌎_Project_Page-DigiFakeAV-blue.svg" height="25">
</a>
<a href="https://huggingface.co/datasets/cambrain/DigiFakeAV/tree/main" target="_blank">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" height="25">
</a>
<a href="https://github.com/DigiFakeAV/DigiFakeAV-v1" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-DigiFakeAV--v1-black?logo=github" height="25">
</a>
<img src="https://img.shields.io/badge/ICASSP-2026-purple" height="25">

</div>

---

## 📢 News
- **2026-03**: Dataset v1.0 is publicly released.
- **2026-01**: DigiFakeAV is accepted to **ICASSP 2026** 🎉

---

## 🧾 TODO List

- [ ] Release DigiFakeAV-v2 with additional generation methods
- [ ] Release DigiShield model weights and inference code
- [ ] Add leaderboard for benchmark evaluation
- [x] Release DigiFakeAV-v1 dataset
- [x] Release processing pipeline code

---

## 🔥 Highlights

- **Large-Scale Dataset:** DigiFakeAV contains **60,000 videos (8.4 million frames)** with diverse identities across nationalities, skin tones, and genders.
- **Diffusion-Based Forgery:** The first large-scale multimodal deepfake dataset built entirely upon **diffusion-based digital human synthesis**, going beyond traditional face-swapping.
- **Three Subset Categories:**
  `RV-RA` (Real Video–Real Audio),
  `FV-RA` (Fake Video–Real Audio),
  `FV-FA` (Fake Video–Fake Audio).
- **Challenging Benchmark:** Detection models suffer over **30% performance drop** on DigiFakeAV compared to existing datasets.
- **DigiShield Detector:** We propose DigiShield, an audio-visual fusion detection model achieving state-of-the-art performance on DF-TIMIT and establishing a strong baseline on DigiFakeAV.

---

## 📚 Contents

- [📢 News](#-news)
- [🔥 Highlights](#-highlights)
- [📚 Contents](#-contents)
- [📖 Abstract](#-abstract)
- [📦 Dataset Overview](#-dataset-overview)
- [🎬 Data Collection](#-data-collection)
- [🧪 Data Synthesis](#-data-synthesis)
- [⚙️ Processing Pipeline](#️-processing-pipeline)
- [🤖 Generation Methods](#-generation-methods)
- [🛠️ Installation](#️-installation)
- [📂 Data Preparation & Reconstruction](#-data-preparation--reconstruction)
- [🔭 Future Works](#-future-works)
- [😄 Acknowledgement](#-acknowledgement)
- [📜 Citation](#-citation)

---

## 📖 Abstract

> In recent years, deepfake technology has advanced rapidly, yet its misuse poses serious threats to information security and public safety. Existing datasets have mainly focused on traditional face-swapping techniques, failing to reflect the emerging trend of digital human generation methods. These diffusion-based approaches can generate highly realistic videos from speech and target images, offering greater flexibility, stealthiness, and multimodal coherence, thus challenging current detection strategies.
>
> To address this issue, we introduce DigiFakeAV, the first large-scale multimodal deepfake dataset based on digital human synthesis. It contains 60,000 videos (8.4 million frames) with diverse identities across nationalities, skin tones, and genders. Experimental results show that state-of-the-art detection models suffer over 30% performance drop on DigiFakeAV, and user studies confirm that the fake videos are nearly indistinguishable from real ones. Furthermore, we propose DigiShield, an audio-visual fusion detection model that achieves state-of-the-art performance on DF-TIMIT and establishes a benchmark for DigiFakeAV. This work presents the first systematic effort in constructing and evaluating a dataset tailored for diffusion-based digital human forgery, highlighting new research directions for robust deepfake detection.

---

## 📦 Dataset Overview

| Subset | Video Type | Audio Type | # Videos | Generation Methods |
|:------:|:----------:|:----------:|:--------:|:------------------:|
| **RV-RA** | Real | Real | 10,000 | — |
| **FV-RA** | Fake | Real | 25,000 | Sonic, Hallo, Hallo2, EchoMimic, V-Express |
| **FV-FA** | Fake | Fake | 25,000 | Sonic, Hallo, Hallo2, EchoMimic + CosyVoice 2 |
| **Total** | — | — | **60,000** | — |

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/figure03.png" width="80%"/>
</div>

---

## 🎬 Data Collection

### Real Video – Real Audio (RV-RA)

From a pool of nearly 40,000 clean video clips, we selected **10,000 real videos** representing diverse ethnicities, genders, and ages. These videos were further processed to crop the upper body region, forming the basis of our real video dataset.

<table align="center">
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_1887%2000_00_00-00_00_30.gif" width="180"/>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_19%2000_00_00-00_00_30.gif" width="180"/>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_8%2000_00_00-00_00_30.gif" width="180"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9407%2000_00_00-00_00_30.gif" width="180"/>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9822%2000_00_00-00_00_30.gif" width="180"/>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9995%2000_00_00-00_00_30.gif" width="180"/>
    </td>
  </tr>
</table>

---

## 🧪 Data Synthesis

### Fake Video – Real Audio (FV-RA)

This category consists of fake videos created by synthesizing visual content conditioned on **authentic audio**. We begin by extracting audio from real videos and converting it into WAV format. RetinaFace is then employed to detect and crop representative facial frames. Using five digital human generation techniques — **Sonic, Hallo, Hallo2, EchoMimic, and V-Express** — we produce a total of **25,000 forged videos**.

Modern adversaries can obtain both voice and facial data of targets, enabling highly convincing impersonation attacks. Compared to traditional face-swapping methods, these forgeries pose a more credible and serious threat. This subset provides researchers with valuable data to explore advanced identity deception scenarios.

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_10000_real_videos_10000%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_113_real_videos_113%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_2678_real_videos_2678%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_3023_real_videos_3023%2000_00_00-00_00_30.gif" width="180"/>
</div>
<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_3056_real_videos_3056%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_6464_real_videos_6464%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_66_real_videos_66%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_9889_real_videos_9889%2000_00_00-00_00_30.gif" width="180"/>
</div>

---

### Fake Video – Fake Audio (FV-FA)

This category encompasses both **manipulated audio and video**. We employed a state-of-the-art voice cloning method, **CosyVoice 2**. Specifically, we first generated manipulated text using a large language model. Then, by providing authentic audio–manipulated text pairs, the CosyVoice 2 model synthesized fake audio exhibiting the vocal characteristics of the target speaker. Subsequently, using **Sonic, Hallo, Hallo2, and EchoMimic**, we generated **25,000 forged videos** based on this fake audio. This approach offers fraudsters a more flexible and realistic means of deception.

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/FAFV_real_3_FAFV_real_3%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_1898_NEOreal_1898%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4217_NEOreal_4217%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4829_NEOreal_4829%2000_00_00-00_00_30.gif" width="180"/>
</div>
<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4935_NEOreal_4935%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_855%20(4)%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_895%20(5)%2000_00_00-00_00_30.gif" width="180"/>
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_954%20(5)%2000_00_00-00_00_30.gif" width="180"/>
</div>

---

## ⚙️ Processing Pipeline

We implemented a dedicated video processing pipeline to segment all MP4, AVI, and MKV files from the original directories into shorter clips, followed by cropping and re-encoding.

Specifically, for each video file:
1. We first calculated its **total duration** and split it into segments of **5 seconds** in length.
2. Any remaining segment with a duration between **3 and 5 seconds** was retained.
3. If the remaining duration was between **2 and 3 seconds**, it was merged with the previous clip to ensure a minimum length of 1 second.
4. Each segment was processed using **FFmpeg**, including trimming from specified start times, resizing to a resolution of **512×512 pixels**, copying the audio stream without re-encoding, and setting the bitrate to **3000k** to balance processing speed and quality.

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/figure03.png" width="85%"/>
</div>

---

## 🤖 Generation Methods

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/QQ20250513-153623.png" width="85%"/>
</div>

| Method | Key Technique | Strength |
|:------:|:-------------:|:--------:|
| **V-Express** | Progressive signal dropping with multimodal condition balancing | Coordinated pose, image, and audio-driven generation |
| **Sonic** | Time-aware fusion + long-range audio context | Continuous and natural head motion synthesis |
| **EchoMimic** | Hybrid audio & facial keypoint conditioning + regional mapping | Natural facial expressions with stable audio-visual sync |
| **Hallo** | Latent diffusion + hierarchical multi-level cross-attention | Fine-grained audio-visual alignment |
| **Hallo2** | VQ-based encoding + temporal alignment mechanism | Long-duration, ultra-high-resolution video synthesis |
| **CosyVoice 2** | Multi-stage semantic decoding + conditional flow matching | Efficient streaming/non-streaming speech synthesis |

### Detailed Descriptions

**V-Express**
By progressively dropping strong signals, the proposed strategy effectively balances multimodal conditions, enabling weak signals to gradually participate in control, thereby achieving coordinated pose, image, and audio-driven generation of highly synchronized deepfake videos.

**Sonic**
Sonic employs time-aware fusion across video segments to disentangle head motion from facial expressions. By leveraging long-range audio context, it achieves continuous and natural motion synthesis, enabling the generation of coherent and realistic deepfake videos.

**EchoMimic**
By integrating hybrid conditioning of audio and facial keypoints, and employing regional mapping alongside motion synchronization strategies, this method generates deepfake videos featuring highly natural facial expressions and stable audio-visual synchronization.

**Hallo**
A latent diffusion model with hierarchical audio-driven visual synthesis is employed, combined with a multi-level cross-attention mechanism to achieve finer audio-visual alignment, thereby enhancing the continuity and realism of the synthesized videos.

**Hallo2**
Hallo2 is optimized for long-duration and ultra-high-resolution video generation. Employing vector quantization (VQ)-based encoding combined with a temporal alignment mechanism, it enables stable synthesis of long-form deepfake videos.

**CosyVoice 2**
By employing multi-stage semantic decoding and conditional flow matching techniques, along with a unified streaming and non-streaming language model design, this approach provides an efficient and stable foundation for speech synthesis in audio-driven digital human videos.

