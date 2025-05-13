# DigiFakeAV-v1
![figure01](https://github.com/DigiFakeAV/DigiFakeAV-v1/blob/main/assets/figure01.png)
The code and data used in this study are publicly available in this repository for the following paper:

**Beyond Face Swapping: DigiFakeAV, a Digital Human-Based Deepfake Video Dataset for Robust Multimodal Detection**
[**Project Page**](https://liming-jiang.com/projects/DrF1/DrF1.html) |   [**Paper**](https://arxiv.org/abs/2001.03024) | [**YouTube Demo**](https://www.youtube.com/watch?v=b6iKqkJht38)

> **Abstract:** *In recent years, deepfake technology has advanced rapidly, yet its misuse poses serious threats to information security and public safety. Existing datasets have mainly focused on traditional face-swapping techniques, failing to reflect the emerging trend of digital human generation methods. These diffusion-based approaches can generate highly realistic videos from speech and target images, offering greater flexibility, stealthiness, and multimodal coherence, thus challenging current detection strategies.
> To address this issue, we introduce DigiFakeAV , the first large-scale multimodal deepfake dataset based on digital human synthesis. It contains 60,000 videos (8.4 million frames) with diverse identities across nationalities, skin tones, and genders. Experimental results show that state-of-the-art detection models suffer over 30\% performance drop on DigiFakeAV, and user studies confirm that the fake videos are nearly indistinguishable from real ones. Furthermore, we propose AVTSF , an audio-visual fusion detection model that achieves state-of-the-art performance on DF-TIMIT and establishes a benchmark for DigiFakeAV. This work presents the first systematic effort in constructing and evaluating a dataset tailored for diffusion-based digital human forgery, highlighting new research directions for robust deepfake detection.*

## Data Collection
From the pool of nearly 40,000 clean video clips, we selected 10,000 real videos representing diverse ethnicities, genders, and ages. These videos were further processed to crop the upper body region, forming the basis of our real video dataset.
<div style="display: flex; flex-wrap: wrap; gap: 10px;">

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/real_videos_1887%2000_00_00-00_00_30.gif" alt="gif1" width="30%" />

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/your_gif2.gif" alt="gif2" width="30%" />

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/your_gif3.gif" alt="gif3" width="30%" />

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/your_gif4.gif" alt="gif4" width="30%" />

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/your_gif5.gif" alt="gif5" width="30%" />

  <img src="https://raw.githubusercontent.com/DigiFakeAV/DigiFakeAV-v1/main/assets/your_gif6.gif" alt="gif6" width="30%" />

</div>
