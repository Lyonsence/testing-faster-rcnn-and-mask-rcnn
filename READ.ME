目标检测与实例分割代码库 (Faster R-CNN & Mask R-CNN)
声明：本仓库仅作为个人学习笔记与代码备份，仅供简单代码存储和参考用途。所有代码均基于开源项目修改，非官方实现，不保证通用性和性能。使用前请自行测试和修改。

📁 项目结构
text
.
├── faster-rcnn.pytorch/     # Faster R-CNN 复现项目（基于 jwyang 实现）
│   ├── data/                # 数据集存放目录（需自行下载）
│   ├── lib/                 # 核心库（包含自定义层、工具函数）
│   ├── models/              # 训练保存的模型
│   ├── trainval_net.py      # 训练脚本
│   ├── test_net.py          # 测试脚本
│   ├── demo.py              # 单图推理可视化
│   └── ...                  # 配置文件等
├── mask-rcnn.pytorch/       # Mask R-CNN 复现项目（基于 TorchVision 官方 API）
│   ├── data/                # COCO 或 Penn-Fudan 数据集
│   ├── models/              # 训练保存的模型（按数据集分子目录）
│   ├── outputs/             # 推理输出图像
│   ├── my_dataset.py        # Penn-Fudan 数据集加载器
│   ├── coco_dataset.py      # COCO 数据集加载器
│   ├── train.py             # Penn-Fudan 训练脚本
│   ├── train_coco.py        # COCO 训练脚本（支持混合精度、断点续训）
│   ├── predict.py           # Penn-Fudan 单图推理
│   ├── batch_infer_torchvision.py  # COCO 批量推理（按类别颜色可视化）
│   ├── evaluate_all.py      # Penn-Fudan 全量评估（计算 mAP）
│   └── ...                  # 其他辅助脚本
└── .gitignore               # Git 忽略文件
📦 数据集说明
Faster R-CNN 使用的数据集
PASCAL VOC 2007 / 2012：标准目标检测数据集，包含 20 个类别。

下载地址：http://host.robots.ox.ac.uk/pascal/VOC/

数据应放置于 faster-rcnn.pytorch/data/VOCdevkit/ 下，目录结构为：

text
VOCdevkit/
├── VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   └── ImageSets/
└── VOC2012/ (可选)
Mask R-CNN 使用的数据集
Penn-Fudan 行人数据集：170 张图像，用于单类别实例分割。下载后解压至 mask-rcnn.pytorch/pennfudan_data/PennFudanPed。

COCO 2017：80 个类别的大规模实例分割数据集。在 AutoDL 平台可通过公共目录获取，或从官网下载。数据应置于 mask-rcnn.pytorch/data/ 下：

text
data/
├── train2017/
├── val2017/
└── annotations/
🔗 代码来源与致谢
Faster R-CNN 部分：基于 jwyang/faster-rcnn.pytorch（PyTorch 1.0 分支）修改，适配了更现代的 PyTorch 版本并修复了部分兼容性问题。

Mask R-CNN 部分：使用 PyTorch TorchVision 官方 API (torchvision.models.detection.maskrcnn_resnet50_fpn)，数据加载和训练脚本参考了官方教程并根据个人需求进行了重构。

⚙️ 使用方式
环境配置（推荐使用 Conda）
bash
# Faster R-CNN 环境（Python 3.8）
conda create -n fasterrcnn python=3.8
conda activate fasterrcnn
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r faster-rcnn.pytorch/requirements.txt

# Mask R-CNN 环境（Python 3.8）
conda create -n maskrcnn python=3.8
conda activate maskrcnn
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python pycocotools matplotlib pillow
Faster R-CNN 训练与测试
bash
# 进入项目目录
cd faster-rcnn.pytorch

# 下载预训练骨干网络权重（ResNet-101）
mkdir -p data/pretrained_model
wget -P data/pretrained_model https://download.openmmlab.com/pretrain/third_party/resnet101_caffe-3ad79236.pth

# 训练（以 VOC 2007 为例）
python trainval_net.py --dataset pascal_voc --net res101 --bs 2 --lr 0.001 --epochs 10 --cuda

# 测试
python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 7 --checkpoint 10021 --cuda

# 单图推理
python demo.py --net res101 --checksession 1 --checkepoch 7 --checkpoint 10021 --cuda --image_dir demo_images
Mask R-CNN 训练与推理（Penn-Fudan）
bash
cd mask-rcnn.pytorch

# 训练
python train.py

# 单图推理
python predict.py

# 全量评估
python evaluate_all.py
Mask R-CNN 训练与推理（COCO）
bash
cd mask-rcnn.pytorch

# 训练（自动保存 checkpoint 和模型权重）
python train_coco.py --batch_size 4 --epochs 5 --lr 0.01 --num_workers 4

# 批量推理（按类别着色）
python batch_infer_torchvision.py --input_dir /path/to/test_images --output_dir /path/to/results --model_path models/coco/model_epoch5.pth --score_thresh 0.5

# 从断点恢复训练
python train_coco.py --resume models/coco/checkpoint_epoch3.pth --epochs 5
📄 注意事项
本仓库代码仅在 AutoDL 云服务器（RTX 4090D，CUDA 11.3）上测试通过，不保证在其他环境下无错运行。

所有模型权重和数据集请自行下载，仓库不包含任何数据文件。

训练超参数（如 batch_size、学习率）可能需要根据您的 GPU 显存进行调整。

遇到问题时，可参考历史对话中的常见问题解决方案（如过滤无效边界框、修改 transforms 等）。

📜 许可
本仓库仅用于个人学习记录，代码遵循各自原始项目的开源协议。请勿用于商业用途。

如有疑问，欢迎提 Issue。
Happy Coding! 🚀