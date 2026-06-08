import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
from my_dataset import PennFudanDataset
import time
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json
import tempfile

# 创建模型保存目录
os.makedirs("models", exist_ok=True)

def get_transform(train):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # 替换掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def print_matched_layers(model):
    tmp_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    pretrained_state = tmp_model.state_dict()
    model_state = model.state_dict()
    total = len(model_state)
    matched = sum(1 for k in model_state if k in pretrained_state and model_state[k].shape == pretrained_state[k].shape)
    print(f"Pretrained weights loaded successfully (strict=False). Matched {matched} layers.")
    del tmp_model

def compute_mAP(model, dataloader, device, iou_threshold=0.5):
    """计算模型在给定 dataloader 上的 mAP（简化版，基于 IoU 和精度/召回）"""
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    all_scores = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # 真实框
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                if len(gt_boxes) == 0:
                    continue
                # 预测框
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                # 过滤低分检测（阈值 0.5）
                keep = pred_scores > 0.5
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                all_gt_boxes.append(gt_boxes)
                all_pred_boxes.append(pred_boxes)
                all_scores.append(pred_scores)

    # 计算所有图片的平均精度（简单平均，实际应用建议用 pycocotools）
    ap_sum = 0.0
    num_images = len(all_gt_boxes)
    for gt, pred, scores in zip(all_gt_boxes, all_pred_boxes, all_scores):
        if len(pred) == 0:
            continue
        # 计算 IoU
        ious = torchvision.ops.box_iou(torch.as_tensor(pred), torch.as_tensor(gt))
        # 对每个预测框，找最大 IoU 的 GT
        max_iou, _ = ious.max(dim=1)
        tp = (max_iou >= iou_threshold).float()
        if tp.sum() == 0:
            continue
        # 按得分排序
        sorted_idx = torch.argsort(torch.as_tensor(scores), descending=True)
        tp = tp[sorted_idx]
        # 累积精度
        prec = tp.cumsum(0) / (torch.arange(1, len(tp)+1).float())
        ap = prec.mean().item()
        ap_sum += ap
    return ap_sum / num_images if num_images > 0 else 0.0

def compute_coco_map(model, dataloader, device):
    """使用 COCO API 计算 mAP（更可靠）"""
    model.eval()
    images_info = []
    annotations = []
    ann_id = 1
    results = []

    with torch.no_grad():
        for img_id, (images, targets) in enumerate(dataloader):
            # 注意：dataloader 返回的 batch 中 images 是列表（长度为 batch_size）
            # 这里简化：假设 batch_size=1，避免复杂转换
            if len(images) != 1:
                raise ValueError("compute_coco_map requires batch_size=1 for simplicity")
            img = images[0].to(device)
            target = targets[0]

            # 真实标注
            h, w = img.shape[1], img.shape[2]
            images_info.append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": f"image_{img_id}.jpg"
            })
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "area": float(width * height),
                    "iscrowd": 0
                })
                ann_id += 1

            # 预测
            pred = model([img])[0]
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            for box, score in zip(pred_boxes, pred_scores):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                results.append({
                    "image_id": img_id,
                    "category_id": 1,   # person
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score)
                })

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_gt:
        json.dump({"images": images_info, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}, f_gt)
        gt_file = f_gt.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_dt:
        json.dump(results, f_dt)
        dt_file = f_dt.name

    # 评估
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_val = coco_eval.stats[0]  # mAP@0.5:0.95

    # 清理临时文件
    os.unlink(gt_file)
    os.unlink(dt_file)
    return map_val


def main():
    # ---------- 数据集路径 ----------
    data_root = "/root/autodl-tmp/mask-rcnn.pytorch/pennfudan_data/PennFudanPed"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    model_dir = "models/pennfudan"
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 数据准备 ----------
    print("Preparing training data...")
    full_dataset = PennFudanDataset(data_root, get_transform(train=True))
    
    # 划分训练集和验证集（80% 训练，20% 验证）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"done\nTraining set: {len(train_dataset)} images, Validation set: {len(val_dataset)} images")

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ---------- 模型 ----------
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    print_matched_layers(model)

    # ---------- 优化器 ----------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # ---------- 训练参数 ----------
    num_epochs = 20          # 可以适当增加
    session = 1
    total_iter_per_epoch = len(train_loader)
    print_interval = 50
    best_map = 0.0

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        iter_start_time = time.time()
        for iter_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            epoch_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (iter_idx + 1) % print_interval == 0 or (iter_idx + 1) == total_iter_per_epoch:
                time_cost = time.time() - iter_start_time
                lr = optimizer.param_groups[0]['lr']
                # 提取子损失
                rpn_cls = loss_dict['loss_objectness'].item()
                rpn_box = loss_dict['loss_rpn_box_reg'].item()
                rcnn_cls = loss_dict['loss_classifier'].item()
                rcnn_box = loss_dict['loss_box_reg'].item()
                mask_loss = loss_dict['loss_mask'].item()
                # 统计 fg/bg（近似）
                total_boxes = sum(len(t['boxes']) for t in targets)
                total_fg = sum((t['labels'] > 0).sum().item() for t in targets)
                total_bg = total_boxes - total_fg
                print(f"[session {session}][epoch {epoch}][iter {iter_idx+1}/{total_iter_per_epoch}] loss: {total_loss.item():.4f}, lr: {lr:.2e}")
                print(f"\t\t\tfg/bg=({total_fg}/{total_bg}), time cost: {time_cost:.2f}s")
                print(f"\t\t\trpn_cls: {rpn_cls:.4f}, rpn_box: {rpn_box:.4f}, rcnn_cls: {rcnn_cls:.4f}, rcnn_box: {rcnn_box:.4f}, mask_loss: {mask_loss:.4f}")
                iter_start_time = time.time()

        avg_train_loss = epoch_loss / total_iter_per_epoch

        # 验证阶段
        # 验证阶段（仅计算 mAP）
        model.eval()
        val_map = compute_coco_map(model, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val mAP: {val_map:.4f}")

        # 保存最佳模型
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), os.path.join(model_dir, "best_mask_rcnn.pth"))
            print(f"Best model saved with mAP = {best_map:.4f}")

        # 定期保存检查点
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_mask_rcnn.pth"))

    # 最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, "best_mask_rcnn.pth"))
    print(f"训练完成，最终模型已保存为 {os.path.join(model_dir, 'mask_rcnn_pennfudan.pth')}")

if __name__ == "__main__":
    main()