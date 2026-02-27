import os
import math
import argparse
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils.preprocess import read_split_data
from utils.DataSet import MyDataset
from utils.train_epoch import train_one_epoch, evaluate

from models.DigiShield import DigiShield


def main(args):
    if args.device.lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_ids = None
        print("🖥️ Training using the CPU")
    else:
        gpu_ids = [int(id) for id in args.device.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"🚀 Using GPU devices: {gpu_ids}")

    random.seed(42)
    torch.manual_seed(42)

    os.makedirs(args.weights, exist_ok=True)
    if not os.path.exists(args.weights):
        os.makedirs(args.weights)

    train_video_paths, train_labels, val_video_paths, val_labels = read_split_data("")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = MyDataset(
        video_paths=train_video_paths,
        labels=train_labels,
        num_frames=30,
    )
    val_dataset = MyDataset(
        video_paths=val_video_paths,
        labels=val_labels,
        num_frames=30,
    )

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'🧠 Using {nw} data loading processes')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=MyDataset.collate_fn,
        pin_memory=True,
        num_workers=nw,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=MyDataset.collate_fn,
        pin_memory=True,
        num_workers=nw,
        shuffle=False,
        drop_last=True
    )

    print("Run to this point")

    model = DigiShield()
    
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    best_val_acc = 0.0  

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_weight = torch.nn.Parameter(torch.ones(1)).to(device)


    tb_writer = SummaryWriter(log_dir=os.path.join(args.weights, "logs"))

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.weights, f"{args.model_name}_best_DigiShield.pth"))
            print(f"💾 Save the best model (validation accuracy: {val_acc:.2%})")

    tb_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.001)
    parser.add_argument('--video_data-path', type=str, 
                       default="")
    parser.add_argument('--audio_data-path', type=str,
                       default="")
    parser.add_argument('--model-name', default='DigiShield', help='create model name')
    parser.add_argument('--weights', type=str, default='/weights',
                       help='initial weights path')
    parser.add_argument('--device', default='0,1,2,3,4', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
