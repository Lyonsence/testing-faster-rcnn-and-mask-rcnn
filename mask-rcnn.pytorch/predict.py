import torch
import cv2
import numpy as np
import os
from PIL import Image
from my_dataset import PennFudanDataset
from train import get_transform, get_model_instance_segmentation

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2

    # 确认模型文件存在
    model_path = "models/pennfudan/mask_rcnn_pennfudan.pth"
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在！请检查路径。")
        return

    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 测试图片路径
    img_path = "/root/autodl-tmp/mask-rcnn.pytorch/pennfudan_data/PennFudanPed/PNGImages/FudanPed00001.png"
    if not os.path.exists(img_path):
        print(f"错误：图片文件 {img_path} 不存在！")
        return

    img = Image.open(img_path).convert("RGB")
    transform = get_transform(train=False)
    x = transform(img).to(device)

    with torch.no_grad():
        prediction = model([x])[0]

    # 可视化
    img_np = np.array(img)
    masks = prediction['masks'].cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    print(f"检测到 {len(scores)} 个目标")
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if score < 0.5:
            continue
        print(f"目标 {i}: 置信度 {score:.4f}, 边界框 {box}")
        mask_np = mask[0] > 0.5
        img_np[mask_np] = img_np[mask_np] * 0.5 + np.array([0, 255, 0]) * 0.5
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 创建输出目录（如果不存在）
    output_path = "/root/autodl-tmp/mask-rcnn.pytorch/outputs/pennfudan/result.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print(f"检测结果已保存为 {output_path}")

if __name__ == "__main__":
    main()