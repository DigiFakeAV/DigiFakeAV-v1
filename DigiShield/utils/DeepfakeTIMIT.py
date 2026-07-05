import os
import cv2
import librosa
import numpy as np
from tqdm import tqdm
import shutil

# ====== 配置參數 ======
input_root = "/home/Userlist/wangjia/PA/DeepfakeTIMIT/lower_quality"
output_root = "/home/Userlist/wangjia/PA/CongYu/DF-TIMIT-LQ"
target_frames = 20
n_mels = 256
sample_rate = 22050
spec_size = (224, 224)  # 新增頻譜圖尺寸參數

# ====== 路徑清理與重建 ======
def setup_dirs():
    # 清空原有測試數據
    test_fake_dir = os.path.join(output_root, "test", "fake") 
    if os.path.exists(test_fake_dir):
        shutil.rmtree(test_fake_dir)
    
    # 重建標準結構
    os.makedirs(os.path.join(output_root, "test", "fake"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "test", "real"), exist_ok=True)

# ====== 音視頻關聯檢測 ======
def find_audio_pair(video_path):
    """強化版音頻文件匹配"""
    base_name = os.path.basename(video_path)
    # 支持多種命名格式：
    # 1. "si1099-video-mwbt0.avi" -> "si1099.wav"
    # 2. "faks0-sx359-video-fdac1.avi" -> "faks0-sx359.wav"
    parts = base_name.split('-')
    
    # 情況1：含video關鍵字
    if 'video' in parts:
        audio_stem = parts[0]
    # 情況2：長格式文件名
    else:
        audio_stem = '-'.join(parts[:2]) if len(parts) > 2 else parts[0]
    
    parent_dir = os.path.dirname(video_path)
    # 掃描所有可能匹配的wav文件
    for f in os.listdir(parent_dir):
        if f.lower().endswith('.wav') and f.startswith(audio_stem):
            return os.path.join(parent_dir, f)
    return None

# ====== 梅爾頻譜可視化 ======
def save_mel_as_jpg(S, save_path):
    """將梅爾頻譜轉換為JET色譜並保存為JPG"""
    # 轉換為分貝單位
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 歸一化並應用JET色譜
    normalized = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # 調整尺寸並保存
    resized = cv2.resize(colored, spec_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, resized)

# ====== 核心處理函數 ======
def process_video(video_path, output_dir):
    """單個視頻處理管道"""
    try:
        # 創建輸出目錄
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(output_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # ====== 視頻幀抽取 ======
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 計算等間隔采樣
        interval = max(total_frames // target_frames, 1)
        frame_indices = [i * interval for i in range(target_frames)]
        frame_indices = [min(idx, total_frames-1) for idx in frame_indices]
        
        # 捕獲並保存幀
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))  # 確保幀尺寸一致
                cv2.imwrite(os.path.join(save_dir, f"frame_{i+1:03d}.jpg"), frame)
        cap.release()
        
        # ====== 音頻處理 ======
        audio_path = find_audio_pair(video_path)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"音頻文件匹配失敗: {video_path}")
        
        # 加載並處理音頻
        y, sr = librosa.load(audio_path, sr=sample_rate)
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        
        # 生成並保存梅爾頻譜圖
        S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels)
        save_mel_as_jpg(S, os.path.join(save_dir, "mel_spec.jpg"))
            
    except Exception as e:
        print(f"\n❌ 處理失敗: {video_path}\n錯誤信息: {str(e)}")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        return False
    
    return True

# ====== 主執行流程 ======
def main():
    setup_dirs()
    
    # 掃描所有AVI視頻
    video_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith('.avi'):
                video_paths.append(os.path.join(root, file))
    
    # 進度條處理
    success = 0
    for path in tqdm(video_paths, desc="🔥 深度偽造數據轉換", unit="video"):
        if process_video(path, os.path.join(output_root, "test", "fake")):
            success += 1
    
    print(f"\n✅ 轉換完成 | 成功: {success}/{len(video_paths)}")

if __name__ == "__main__":
    main()
