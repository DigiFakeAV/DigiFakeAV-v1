# DigiFakeAV-v1
![figure01](https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/figure01.png)
The code and data used in this study are publicly available in this repository for the following paper:

**Beyond Face Swapping: DigiFakeAV, a Digital Human-Based Deepfake Video Dataset for Robust Multimodal Detection**
[**Project Page**](https://liming-jiang.com/projects/DrF1/DrF1.html) |   [**Paper**](https://arxiv.org/abs/2001.03024) | [**YouTube Demo**](https://www.youtube.com/watch?v=b6iKqkJht38)

> **Abstract:** *In recent years, deepfake technology has advanced rapidly, yet its misuse poses serious threats to information security and public safety. Existing datasets have mainly focused on traditional face-swapping techniques, failing to reflect the emerging trend of digital human generation methods. These diffusion-based approaches can generate highly realistic videos from speech and target images, offering greater flexibility, stealthiness, and multimodal coherence, thus challenging current detection strategies.
> To address this issue, we introduce DigiFakeAV , the first large-scale multimodal deepfake dataset based on digital human synthesis. It contains 60,000 videos (8.4 million frames) with diverse identities across nationalities, skin tones, and genders. Experimental results show that state-of-the-art detection models suffer over 30\% performance drop on DigiFakeAV, and user studies confirm that the fake videos are nearly indistinguishable from real ones. Furthermore, we propose AVTSF , an audio-visual fusion detection model that achieves state-of-the-art performance on DF-TIMIT and establishes a benchmark for DigiFakeAV. This work presents the first systematic effort in constructing and evaluating a dataset tailored for diffusion-based digital human forgery, highlighting new research directions for robust deepfake detection.*

## Data Collection
**Real Video–Real Audio (RV-RA)** From the pool of nearly 40,000 clean video clips, we selected 10,000 real videos representing diverse ethnicities, genders, and ages. These videos were further processed to crop the upper body region, forming the basis of our real video dataset.
<table align="center" style="border-collapse: separate; border-spacing: 10px;">
  <tr>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_1887%2000_00_00-00_00_30.gif" alt="gif1" width="200"/></td>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_19%2000_00_00-00_00_30.gif" alt="gif2" width="200"/></td>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_8%2000_00_00-00_00_30.gif" alt="gif3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9407%2000_00_00-00_00_30.gif" alt="gif4" width="200"/></td>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9822%2000_00_00-00_00_30.gif" alt="gif5" width="200"/></td>
    <td><img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_9995%2000_00_00-00_00_30.gif" alt="gif6" width="200"/></td>
  </tr>
</table>

**Fake Video–Real Audio (FV-RA)** This category consists of fake videos created by synthesizing visual content conditioned on authentic audio. We begin by extracting audio from real videos and converting it into WAV format. RetinaFace is then employed to detect and crop representative facial frames. Using five digital human generation techniques—Sonic, Hallo1, Hallo2, Echomimic, and V-Express—we produce a total of 25,000 forged videos. Modern adversaries can obtain both voice and facial data of targets, enabling highly convincing impersonation attacks. Compared to traditional face-swapping methods, these forgeries pose a more credible and serious threat. This subset provides researchers with valuable data to explore advanced identity deception scenarios.
<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_10000_real_videos_10000%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_113_real_videos_113%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_2678_real_videos_2678%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_3023_real_videos_3023%2000_00_00-00_00_30.gif" width="200" />
</div>

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_3056_real_videos_3056%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_6464_real_videos_6464%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_66_real_videos_66%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVRA/real_videos_9889_real_videos_9889%2000_00_00-00_00_30.gif" width="200" />
</div>

**Fake Video–Fake Audio (FV-FA)** This category encompasses both manipulated audio and video. We employed a state-of-the-art voice cloning method, CosyVoice 2. Specifically, we first generated manipulated text using a large language model. Then, by providing authentic audio–manipulated text pairs, the CosyVoice 2 model synthesized fake audio exhibiting the vocal characteristics of the target speaker. Subsequently, using Sonic, Hallo1, Hallo2, and Echomimic, we generated 25,000 forged videos based on this fake audio. This approach offers fraudsters a more flexible and realistic means of deception.
<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/FAFV_real_3_FAFV_real_3%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_1898_NEOreal_1898%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4217_NEOreal_4217%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4829_NEOreal_4829%2000_00_00-00_00_30.gif" width="200" />
</div>

<div align="center">
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/NEOreal_4935_NEOreal_4935%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_855%20(4)%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_895%20(5)%2000_00_00-00_00_30.gif" width="200" />
  <img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/FVFA/video_954%20(5)%2000_00_00-00_00_30.gif" width="200" />
</div>

## Benchmark

We benchmark nine representative forgery detection methods using the DigiFakeAV dataset，These methods encompass six distinct cate-gories: Basic (Naive), Spatial, Temporal, Frequency, Hybrid Domain, and Multi-modal approaches. Please refer to our paper for more information.
<img src="https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/QQ20250513-153623.png" />

