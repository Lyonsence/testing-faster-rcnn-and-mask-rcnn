# validate_coco_per_class.py
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from coco_dataset import COCOMaskDataset
from train_coco import get_model_instance_segmentation
import json
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_img_dir = "/root/autodl-tmp/mask-rcnn.pytorch/data/val2017"
    val_ann_file = "/root/autodl-tmp/mask-rcnn.pytorch/data/annotations/instances_val2017.json"
    model_path = "models/coco/model_epoch10.pth"   # 改为您的模型路径
    num_classes = 91

    # 数据预处理：仅 ToTensor（与训练时一致）
    transform = T.Compose([T.ToTensor()])

    # 加载验证集（可限制数量）
    dataset = COCOMaskDataset(val_img_dir, val_ann_file, transforms=transform)
    eval_limit = 2000   # 评估前 2000 张（可改为 None 评估全部）
    if eval_limit and eval_limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(eval_limit))
    print(f"Total validation images: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=lambda x: tuple(zip(*x)))

    # 加载模型
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    results = []
    total_imgs = len(loader)
    print_interval = max(100, total_imgs // 20)

    with torch.no_grad():
        for img_idx, (images, targets) in enumerate(loader):
            img = images[0].to(device)          # 已经是 Tensor，形状 (C, H, W)
            real_img_id = targets[0]['image_id'].item()

            # 模型推理（输入需要 batch 维度）
            pred = model(img.unsqueeze(0))[0]

            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < 0.05:   # 低阈值以召回更多检测
                    continue
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                # 确保边界框有效
                if width <= 0 or height <= 0:
                    continue
                results.append({
                    "image_id": real_img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score)
                })

            if (img_idx + 1) % print_interval == 0 or (img_idx + 1) == total_imgs:
                print(f"Processed {img_idx + 1}/{total_imgs} images")

    # COCO 评估
    coco_gt = COCO(val_ann_file)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        dt_file = f.name
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n========== Overall mAP ==========")
    coco_eval.summarize()

    # 可选：输出每个类别的 AP
    precision = coco_eval.eval['precision']
    cat_ids = sorted(coco_gt.cats.keys())
    cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    print("\n========== Per-class AP (IoU=0.50:0.95) ==========")
    for cat_id in cat_ids:
        idx = cat_id_to_idx[cat_id]
        ap = precision[:, :, idx, 0, -1].mean()
        print(f"Category {cat_id:3d} ({coco_gt.cats[cat_id]['name']:20s}) AP = {ap:.4f}")

    os.unlink(dt_file)

if __name__ == "__main__":
    main()