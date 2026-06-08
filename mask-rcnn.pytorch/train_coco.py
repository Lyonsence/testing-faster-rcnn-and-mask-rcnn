import torch
import torchvision
from torch.utils.data import DataLoader
from coco_dataset import COCOMaskDataset, get_transform
import time
import argparse
from torch.cuda.amp import autocast, GradScaler
import os

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    train_img_dir = "/root/autodl-tmp/mask-rcnn.pytorch/data/train2017"
    train_ann_file = "/root/autodl-tmp/mask-rcnn.pytorch/data/annotations/instances_train2017.json"
    val_img_dir = "/root/autodl-tmp/mask-rcnn.pytorch/data/val2017"
    val_ann_file = "/root/autodl-tmp/mask-rcnn.pytorch/data/annotations/instances_val2017.json"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_train = COCOMaskDataset(train_img_dir, train_ann_file, transforms=get_transform(train=True))
    dataset_val = COCOMaskDataset(val_img_dir, val_ann_file, transforms=get_transform(train=False))

    num_classes = dataset_train.get_num_classes()
    print(f"Number of classes (including background): {num_classes}")

    indices = torch.randperm(len(dataset_val)).tolist()
    val_subset = torch.utils.data.Subset(dataset_val, indices[:2000])

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = GradScaler()
    print("AMP (Automatic Mixed Precision) enabled.")

    model_dir = "models/coco"
    os.makedirs(model_dir, exist_ok=True)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    total_iter_per_epoch = len(train_loader)
    print_interval = 100
    session = 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        iter_start_time = time.time()
        for iter_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast():
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

            if (iter_idx + 1) % print_interval == 0 or (iter_idx + 1) == total_iter_per_epoch:
                time_cost = time.time() - iter_start_time
                lr = optimizer.param_groups[0]['lr']
                print(f"[session {session}][epoch {epoch}][iter {iter_idx+1}/{total_iter_per_epoch}] loss: {total_loss.item():.4f}, lr: {lr:.2e}")
                print(f"\t\t\tfg/bg=(-/-), time cost: {time_cost:.2f}s")
                iter_start_time = time.time()

        avg_loss = epoch_loss / total_iter_per_epoch
        print(f"Epoch {epoch}/{args.epochs} finished, average loss: {avg_loss:.4f}")

        lr_scheduler.step()

        # 保存 checkpoint（用于恢复训练）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # 保存独立的模型权重文件（每个 epoch 一个，便于推理）
        model_path = os.path.join(model_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved: {model_path}")

    # 最终模型
    final_model_path = os.path.join(model_dir, "mask_rcnn_coco_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved as {final_model_path}")

if __name__ == "__main__":
    main()