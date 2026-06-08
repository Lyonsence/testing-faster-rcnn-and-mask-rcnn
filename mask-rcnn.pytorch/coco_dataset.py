import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T

class COCOMaskDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        masks = []
        area = []
        iscrowd = []

        for ann in anns:
            # 跳过没有分割标注的实例
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            # 过滤无效边界框（宽度或高度 <= 0）
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])   # COCO 类别 ID 1~80
            mask = self.coco.annToMask(ann)     # 生成二进制掩膜
            masks.append(mask)
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        if len(boxes) == 0:
            # 如果没有有效标注，返回空张量（训练时会跳过，但可能引发警告，建议过滤）
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)   # 只对图像做变换，不对 target
        return img, target

    def __len__(self):
        return len(self.ids)

    def get_num_classes(self):
        """返回 COCO 类别数（包含背景）"""
        cat_ids = self.coco.getCatIds()
        max_cat_id = max(cat_ids) if cat_ids else 80
        return max_cat_id + 1   # 背景 0 占一位

def get_transform(train):
    transforms = [T.ToTensor()]
    # 训练时通常不加随机翻转，因为需要同步更新 boxes/masks（实现较复杂）
    return T.Compose(transforms)