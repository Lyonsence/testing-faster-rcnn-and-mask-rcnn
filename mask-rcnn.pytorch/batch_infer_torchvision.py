import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
from PIL import Image
import os
import argparse

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

PREDEFINED_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0), (0, 255, 128),
    (255, 0, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 255)
]

def get_color(label):
    if 1 <= label <= len(PREDEFINED_COLORS):
        return PREDEFINED_COLORS[label-1]
    else:
        return (0, 255, 0)  # 默认绿色

def get_model(device, model_path, num_classes=91):
    """加载本地训练好的模型权重"""
    # 构建与训练时相同结构的模型（不加载预训练权重）
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # 替换掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    # 如果是完整 checkpoint（包含 'model_state_dict'），则提取权重
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def process_image(model, image_path, device, score_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0]
    img_np = np.array(img)
    masks = pred['masks'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    for mask, box, score, label in zip(masks, boxes, scores, labels):
        if score < score_thresh:
            continue
        color = get_color(label)
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        mask_np = mask[0] > 0.5
        img_np[mask_np] = (img_np[mask_np] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score:.2f}"
        text_x, text_y = x1, y1 - 5
        if text_y < 5:
            text_y = y1 + 15
        cv2.putText(img_np, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='输入图片文件夹')
    parser.add_argument('--output_dir', required=True, help='输出结果文件夹')
    parser.add_argument('--model_path', required=True, help='训练好的模型权重文件路径 (例如 models/coco/model_epoch5.pth)')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--num_classes', type=int, default=91, help='类别数（含背景）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model(device, args.model_path, args.num_classes)
    print(f"Model loaded from {args.model_path}")

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(img_extensions)]
    print(f"Found {len(img_files)} images.")

    for idx, img_file in enumerate(img_files):
        src = os.path.join(args.input_dir, img_file)
        name, ext = os.path.splitext(img_file)
        dst = os.path.join(args.output_dir, f"{name}_det{ext}")
        try:
            result = process_image(model, src, device, args.score_thresh)
            cv2.imwrite(dst, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"[{idx+1}/{len(img_files)}] Processed: {img_file} -> {dst}")
        except Exception as e:
            print(f"[{idx+1}/{len(img_files)}] Failed: {img_file}, error: {e}")
    print("Done.")

if __name__ == "__main__":
    main()