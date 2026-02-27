import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_con = torch.nn.CosineEmbeddingLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    data_loader = tqdm(data_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, batch in enumerate(data_loader):  
        video = batch['visual'].to(device)  
        audio = batch['audio'].to(device)  
        targets = batch['label'].to(device)
        
        optimizer.zero_grad()
        visual_feat, audio_feat, cls_output = model(video, audio)
        
        contrast_target = torch.ones(video.size(0)).to(device)
        loss_contrast = criterion_con(visual_feat, audio_feat, contrast_target)
        loss_cls = criterion_cls(cls_output, targets)
        total_loss = 0.7 * loss_cls + 0.3 * loss_contrast
        
        total_loss.backward()
        optimizer.step()
        
        _, predicted = cls_output.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        data_loader.set_postfix({
            'Loss': total_loss.item(),
            'Acc': f"{100.*correct/total:.2f}%",
            'ClsLoss': loss_cls.item(),
            'ConLoss': loss_contrast.item()
        })
        
    return total_loss.item(), correct/total

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    data_loader = tqdm(data_loader, desc=f'Validate Epoch {epoch}')
    
    for batch_idx, batch in enumerate(data_loader):  
        video = batch['visual'].to(device)
        audio = batch['audio'].to(device)
        targets = batch['label'].to(device)
        
        _, _, cls_output = model(video, audio)
        loss = criterion_cls(cls_output, targets)
        
        _, predicted = cls_output.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()
        
        data_loader.set_postfix({
            'Loss': loss.item(),
            'Acc': f"{100.*correct/total:.2f}%"
        })
    
    return total_loss/len(data_loader), correct/total
