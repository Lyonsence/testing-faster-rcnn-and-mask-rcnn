import torch
import torchvision
from my_dataset import PennFudanDataset
from train import get_transform, get_model_instance_segmentation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from PIL import Image
import cv2

os.makedirs("outputs", exist_ok=True)

def convert_to_coco_format(dataset, predictions):
    """
    将数据集和模型预测结果转换为 COCO 评估格式。
    dataset: PennFudanDataset 实例
    predictions: 模型对所有图片的输出列表，每个元素是 dict (boxes, scores, labels, masks)
    返回 coco_gt (真实标注) 和 coco_dt (预测结果) 的字典。
    """
    images_info = []
    annotations = []
    ann_id = 1
    for img_id, (img, target) in enumerate(dataset):
        # 真实标注
        h, w = img.shape[1], img.shape[2]   # 因为 transform 后是 tensor，shape (C,H,W)
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
                "category_id": int(label),   # 1 for person
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "area": float(width * height),
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt = {"images": images_info, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}

    # 预测结果
    results = []
    for img_id, pred in enumerate(predictions):
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for box, score in zip(boxes, scores):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            results.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score)
            })
    return coco_gt, results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    data_root = "/root/autodl-tmp/mask-rcnn.pytorch/pennfudan_data/PennFudanPed"

    # 加载整个数据集（不使用划分，全部用于评估）
    full_dataset = PennFudanDataset(data_root, get_transform(train=False))
    print(f"Loaded {len(full_dataset)} images for evaluation.")

    # 加载模型
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load("models/best_mask_rcnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # 推理所有图片
    all_predictions = []
    with torch.no_grad():
        for idx in range(len(full_dataset)):
            img, target = full_dataset[idx]   # img 是 tensor，target 是 dict
            # 添加 batch 维度
            img = img.unsqueeze(0).to(device)
            pred = model(img)[0]
            all_predictions.append(pred)
            # 打印每张图片的检测结果（置信度）
            print(f"Image {idx}:")
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            for box, score in zip(boxes, scores):
                if score > 0.5:
                    print(f"  Person detected with confidence {score:.4f}, bbox: {box}")

    # 使用 COCO 评估工具计算 mAP
    coco_gt, coco_dt = convert_to_coco_format(full_dataset, all_predictions)
    # 写入临时 JSON 文件
    with open("temp_gt.json", "w") as f:
        json.dump(coco_gt, f)
    with open("temp_dt.json", "w") as f:
        json.dump(coco_dt, f)
    coco_gt_obj = COCO("temp_gt.json")
    coco_dt_obj = coco_gt_obj.loadRes("temp_dt.json")
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # mAP 存储在 coco_eval.stats[0] 中
    print(f"Overall mAP@0.5:0.95 = {coco_eval.stats[0]:.4f}")

    # 可选：可视化第一张图片并保存
    img0, _ = full_dataset[0]
    img_np = (img0.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    pred0 = all_predictions[0]
    masks = pred0['masks'].cpu().numpy()
    boxes = pred0['boxes'].cpu().numpy()
    scores = pred0['scores'].cpu().numpy()
    for mask, box, score in zip(masks, boxes, scores):
        if score < 0.5:
            continue
        mask_np = mask[0] > 0.5
        img_np[mask_np] = img_np[mask_np] * 0.5 + np.array([0, 255, 0]) * 0.5
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite("outputs/pennfudan/all_images_sample.jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print("Sample visualization saved to outputs/all_images_sample.jpg")

if __name__ == "__main__":
    main()