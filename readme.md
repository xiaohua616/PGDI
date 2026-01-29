# Prior-Guided Deep Inversion for Few-Shot Knowledge Distillation

Official PyTorch implementation of **PGDI**. We propose a few-shot distillation framework that leverages limited labeled samples as **priors** to guide model inversion, ensuring synthetic data closely matches the real distribution.

![PGDI Framework](assets/framework.png)

## 📊 Results

PGDI achieves state-of-the-art performance on **CIFAR-10**, **CIFAR-100**, and **Tiny-ImageNet** benchmarks.

| Dataset | Teacher | Student | PGDI (10-shot) Top-1 Acc |
| :--- | :--- | :--- | :--- |
| **CIFAR-10** | ResNet-34 | ResNet-18 | **95.10%** |
| | VGG-11 | ResNet-18 | **91.32%** |
| | WRN-40-2 | WRN-16-1 | **89.15%** |
| | WRN-40-2 | WRN-40-1 | **93.23%** |
| | WRN-40-2 | WRN-16-2 | **92.83%** |
| **CIFAR-100** | ResNet-34 | ResNet-18 | **77.13%** |
| | VGG-11 | ResNet-18 | **70.97%** |
| | WRN-40-2 | WRN-16-1 | **53.97%** |
| | WRN-40-2 | WRN-40-1 | **69.55%** |
| | WRN-40-2 | WRN-16-2 | **69.40%** |
| **Tiny-ImageNet** | ResNet-34 | ResNet-18 | **64.17%** |

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Models
To reproduce our results, please download pre-trained teacher models from [Dropbox-Models (266 MB)](https://www.dropbox.com/scl/fo/zbdyf9c7ami7ywmd2z6hd/ALLlm4mb4Ba3Q2njRg1xsOw?rlkey=y42mw3kcehtp4l9h7y55vt2d5&e=1&dl=0) and extract them as `checkpoints/pretrained`.

The directory structure should look like this:
```text
checkpoints/
└── pretrained/
    ├── cifar100_wrn40_2.pth
    ├── cifar100_resnet34.pth
    └── ...
```

### 3. Run Distillation
You can run the distillation using the provided script.

**Example: WRN-40-2 → WRN-16-1 on CIFAR-100 (5-shot)**

```bash
python fewshot_kd.py \
    --method PGDI \
    --dataset cifar100 \
    --teacher wrn40_2 \
    --student wrn16_1 \
    --epochs 200 \
    --fewshot_n_per_class 5 \
    --synthesis_batch_size 512 \
    --gpu 0 \
    --save_dir "run/PGDI_experiment"
```
