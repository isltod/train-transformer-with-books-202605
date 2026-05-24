### 역자 추가 코드
import timm

# all_models = timm.list_models()
# vit_models = [model for model in all_models if 'vit' in model]
#
# # 모델 수 카운트 및 출력
# print(f"Total number of ViT models: {len(vit_models)}")
# print("Available ViT models in timm:")
# for model in vit_models:
#     print(model)

# 데이터 디렉토리는 자신의 구글 드라이브 설정에 맞춰서 변경이 필요합니다.
# 실습 데이터셋(https://www.kaggle.com/datasets/jr2ngb/cataractdataset) 용량은 4GB입니다.
# 이 데이터셋을 구글 드라이브에 폴더당 10개씩의 이미지만 올립니다.
# 축소 데이터셋은 도서의 예제 코드에서 제공합니다.
# 폴더명은 normal, cataract, glaucoma, retina_disease 입니다.

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 눈 질환 데이터셋...각 클래스에 10개씩만 있는 축소판...
DATA_DIR = "../../data/dataset/"
# 정상, 백내장, 녹내장, 망막질환?
CLASSES = ["normal", "cataract", "glaucoma", "retina_disease"]
data = []
# 4개 클래스별로...
for class_idx, class_name in enumerate(CLASSES):
    # 각각 경로만들고
    class_dir = os.path.join(DATA_DIR, class_name)
    # 이미지들 이름별로
    for img_name in os.listdir(class_dir):
        # 이미지 경로 만들고 data에 경로와 라벨 넣기
        img_path = os.path.join(class_dir, img_name)
        data.append([img_path, class_idx])

# 실제 이미지가 아니라 경로와 라벨로 데이터셋 만드는데...
df = pd.DataFrame(data, columns=["image_path", "label"])
# 시험 자료는 20%
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# 훈련 이미지 전처리 - 크기 224 이건 필수, 무작위 수평 뒤집기, 텐서로 만들고, 정규화...
# 근데 정규화에 쓰이는 이 평균과 표준편차는 어디서 온거지?
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 시험 이미지도 전처리...수평 뒤집기는 뺀다...정규화 모수는 훈련 거 그대로...
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 백내장만 보나? 아니면 대표 이름인가...
class CataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get은 이미지 경로로 이미지 열고, RGB로...전처리 있으면 하고
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 라벨과 함께 반환
        label = self.labels[idx]
        return img, label


# 데이터프레임에서 데이터셋, 데이터 로더 만들고...
train_dataset = CataractDataset(
    train_df["image_path"].values, train_df["label"].values, transform=train_transforms
)
test_dataset = CataractDataset(
    test_df["image_path"].values, test_df["label"].values, transform=test_transforms
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 여기서 데이터를 보자는데...훈련데이터로...
data_iter = iter(train_loader)
images, labels = next(data_iter)


# 텐서로 만든 이미지를 받아서 역정규화(unnormalize)하고 넘파이로 반환
def imshow(img_tensor):
    # 처음부터 넘파이로 만들어서
    img = img_tensor.numpy()
    # 축을 이렇게 전치해야 matplotlib에서 보이는 모양...
    img = np.transpose(img, (1, 2, 0))
    # 데이터셋으로 만들 때 했던 정규화 되돌리고...
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    # 근데 역정규화하니 오히려 값이 더 0에 가깝게 작아지네?
    img = np.clip(img, 0, 1)
    return img


# 이미지와 레이블 디스플레이
fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
for idx, (image, label) in enumerate(zip(images, labels)):
    axes[idx].imshow(imshow(image))
    axes[idx].set_title(f"Label: {label.item()}")
    axes[idx].axis("off")
plt.show()
