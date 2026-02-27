
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

from utils.preprocess import read_test_data
from utils.DataSet import MyDataset
from models.DigiShield import DigiShield

def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Operating Equipment: {device}")

    model = DigiShield().to(device)
    
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"💾 Model weights loaded successfully: {args.weights}")
    else:
        raise FileNotFoundError(f"❌ The weighting file does not exist: {args.weights}")

    test_paths, test_labels = read_test_data(args.test_root)

    test_dataset = MyDataset(
        video_paths=test_paths,
        labels=test_labels,
        num_frames=20,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count(), 8),
        pin_memory=True,
        collate_fn=MyDataset.collate_fn
    )

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🧪 Testing Progress"):
            videos = batch['visual'].to(device)
            audios = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            _, _, logits = model(videos, audios)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels, dtype=np.int32)
    all_preds = np.array(all_preds, dtype=np.int32)
    
    unique_labels = np.unique(all_labels)
    valid_labels = unique_labels[unique_labels >= 0]  
    
    label_names = ['real', 'fake']
    present_labels = []
    for lbl in valid_labels:
        if lbl < len(label_names):
            present_labels.append(lbl)
    
    if len(present_labels) == 0:
        raise RuntimeError("❌ Invalid tag dataset detected")
    target_names = [label_names[lbl] for lbl in sorted(present_labels)]

    accuracy = np.nanmean(all_preds == all_labels) if len(all_labels) > 0 else 0.0
    
    if len(np.unique(all_labels)) >= 2:
        auc = roc_auc_score(all_labels, all_probs)
        auc_status = f"{auc:.4f}"
    else:
        auc_status = "N/A (At least two categories are required)"
    
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    
    if len(cm) == len(target_names):
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=target_names, 
            digits=4,
            zero_division=0
        )
    else:
        report = "⚠️ Classification report unavailable (label and prediction mismatch)"

    print("\n📈 Performance Evaluation Report:")
    print(f"🔢 Accuracy: {accuracy:.4f}")
    print(f"📊 AUC: {auc_status}")
    print("\n🔀 Confusion Matrix:")
    print(cm)
    print("\n📝 Classification Report:")
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "test_report_DigiShield.txt")
    with open(result_path, "w") as f:
        f.write("🏷️ Deepfake Detection Test Report\n\n")
        f.write(f"🔢 ACC: {accuracy:.4f}\n")
        f.write(f"📊 AUC: {auc_status}\n\n")
        f.write("🔀 Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\n📝 Test Report:\n")
        f.write(report)
    print(f"\n💾 The results have been saved to: {result_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake Detection Test Script')
    parser.add_argument('--test-root', 
                      type=str, 
                      default="",
                      help='Test Set Root Directory Path')
    parser.add_argument('--weights', 
                      type=str, 
                      default="",
                      help='Trained model weight path')
    parser.add_argument('--batch-size', 
                      type=int, 
                      default=16,
                      help='Batch size')
    parser.add_argument('--device', 
                      type=str, 
                      default='4',
                      help='CUDA device ID (e.g., 0) or CPU')
    parser.add_argument('--output-dir', 
                      type=str, 
                      default='./results',
                      help='Output Results Directory')

    args = parser.parse_args()
    main(args)
