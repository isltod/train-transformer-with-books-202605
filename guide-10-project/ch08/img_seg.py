import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from transformers import default_data_collator
from datasets import load_dataset, DatasetDict, load_from_disk
import os
from PIL import Image
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../../")
from wolf import get_my_gpu_device

device = get_my_gpu_device(0)


def process_image(image_file, image_directory, label_directory):
    # 파일 이름에 .png가 라벨이라고...
    file_name, _ = os.path.splitext(image_file)
    label_file = f"{file_name}.png"

    # 원본 이미지 복사본 만들고
    with Image.open(os.path.join(image_directory, image_file)) as im:
        img = im.copy()
    # 라벨 이미지는 8비트 그레이...
    with Image.open(os.path.join(label_directory, label_file)) as im:
        lbl = im.convert("L").copy()
    # 딕셔너리로 만들어 반환
    return {"pixel_values": img, "label": lbl}


def load_images_and_labels(image_directory, label_directory):
    # 이미지 파일 목록
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(".jpg")])
    # 이건 안쓰는데?
    label_files = sorted([f for f in os.listdir(label_directory) if f.endswith(".png")])

    data = []
    # 스레드 작업...이건 IO 작업에 쓰고, CPU에는 ProcessPoolExecutor를 사용하라고...
    with ThreadPoolExecutor() as executor:
        # image_files 이미지들에 process_image 함수 적용...이미지와 라벨 사전들의 리스크인가?
        results = executor.map(
            process_image,
            # 이 아래 3개 리스트 원소들이 process_image 함수에 매개변수로 전달되는데...
            image_files,
            # 각각 다 image_directory와 label_directory를 받아야 하므로 곱하기로 image_files 수 만큼 리스트로...
            [image_directory] * len(image_files),
            [label_directory] * len(image_files),
        )
        # 위 결과가 리스트가 아니라 map 객체로 나온다..그걸 리스트로 만들어 반환
        for result in results:
            data.append(result)

    return data


def create_image_segmentation_dataset(image_directory, label_directory):
    # 이미지와 라벨 딕셔너리들의 리스트 받아서
    data = load_images_and_labels(image_directory, label_directory)
    # 그걸 같은 key를 묶고 그 밑에 리스트로 딕셔너리 만들고, 그걸 Dataset으로 만들어 반환
    dataset = Dataset.from_dict(
        {
            "pixel_values": [item["pixel_values"] for item in data],
            "label": [item["label"] for item in data],
        }
    )

    return dataset


# 학습 이미지, 레이블 이미지, 저장 경로...저장은 허깅페이스 hf 형식으로 저장...
image_directory = "../../data/FoodSeg103_short/Images/img_dir/train"
label_directory = "../../data/FoodSeg103_short/Images/ann_dir/train"
output_path = "../../data/train_dataset.hf"

# 데이터셋 생성
train_dataset = create_image_segmentation_dataset(image_directory, label_directory)

# 허깅페이스 형식으로 Dataset을 저장
train_dataset.save_to_disk(output_path)
print(f"Dataset saved to {output_path}")
# 저장한 데이터셋을 다시 불러온다...왜지? 실습인가?
train_dataset = load_from_disk(output_path)
print(train_dataset)

# 학습과 시험 데이터 분리 - 원본 데이터는 0.98, 축소 데이터는 0.7
# train_test_split_ratio = 0.98
train_test_split_ratio = 0.7
train_size = int(train_test_split_ratio * len(train_dataset))
test_size = len(train_dataset) - train_size
split_ds = train_dataset.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
# 근데 데이터셋을 최종, 훈련, 시험 세 가지로 만들어?
final_ds = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})
train_ds = split_ds["train"]
test_ds = split_ds["test"]

# 각 클래스 번호와 이름으로 상호 변환 딕셔너리로 만들기...
base_dir = "../../data/FoodSeg103_short/"
id2label = {}
with open(base_dir + "category_id.txt", "r") as file:
    for line in file:
        id_, label = line.strip().split("\t")
        id2label[int(id_)] = label
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(id2label)
print(label2id)

# 원본 이미지와 라벨 이미지를 좀 보자...
# 0번 라벨 이미지를 넘파이 배열로
data = train_ds[0]
# Red 채널이란 설명은 틀린거 아닌가? 라벨 이미지는 그레이 스케일로 만들잖아?
r_channel_array = np.array(data["label"])
# 이건 뭐 이미지 클래스들의 집합을 만드는 거겠지...
unique_categories = np.unique(r_channel_array)
print("Unique categories in the image:", unique_categories)

# # 다시 화면으로...원본과 라벨 이미지 표시
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# ax1.imshow(data["pixel_values"])
# ax1.set_title("Image")
# ax1.axis("off")
#
# ax2.imshow(data["label"], cmap="gray")
# ax2.set_title("Segmentation Label")
# ax2.axis("off")
# # 이미지에서 유일한(Unique) 카테고리: [ 0 48 84 85 87] - 이건 뭔 설명이지?
#
# plt.show()

from transformers import SegformerImageProcessor
from torchvision.transforms import ColorJitter

feature_extractor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


# 학습 데이터는 밝기 등을 손대고...
def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    # Segformer 모델이 인식할 수 있도록 크기 조정, 정규화, 실수 변환, 라벨 마스크 처리, 텐서로 등 처리한다고...
    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels)
    return inputs


# 뭔가 데이터셋 이미지들 전처리하는 방법인거 같네...
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

from transformers import SegformerForSemanticSegmentation

# 모델 만들고...
model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name, id2label=id2label, label2id=label2id
)

from transformers import TrainingArguments

epochs = 10
lr = 3e-5
batch_size = 4

training_args = TrainingArguments(
    "/content/drive/MyDrive/Book6/Ch8/run",  # 출력 디렉토리 변경
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=5,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=10000,
    eval_steps=1000,
    logging_steps=500,
    eval_accumulation_steps=10,
    load_best_model_at_end=True,
)

import evaluate
import torch
from torch import nn

# 이 아래는 아무 설명이 없어서 뭔지 모르겠고...
# IoU = (Intersection of predicted mask and ground truth mask) / (Union of predicted mask and ground truth mask)
iou_metric = evaluate.load("mean_iou")


def calculate_segmentation_metrics(prediction_ground_truth):
    with torch.no_grad():
        logits, ground_truth = prediction_ground_truth
        logits_as_tensor = torch.from_numpy(logits)

        # segmentation 모델 학습 시, output(logits)이 ground_truth 레이블에 비교해서
        # 낮은 해상도(spatial resolution)을 갖게 됨
        # 때문에 logits를 조정(resize)해서 ground_truth 레이블 사이즈와 매칭시킴
        resized_logits = nn.functional.interpolate(
            logits_as_tensor,
            size=ground_truth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        # 넘파이로 변환 전에 cpu로 전송
        predicted_labels = resized_logits.detach().cpu().numpy()

        # 평균 IoU(mean IoU) 메트릭(metric) 계산
        segmentation_metrics = iou_metric._compute(
            predictions=predicted_labels,
            references=ground_truth,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.do_reduce_labels,
        )

        # 개별적인 키-값(key-value) 쌍으로서 카테고리당(per-category) 메트릭(metrics) 추출
        category_accuracy = segmentation_metrics.pop("per_category_accuracy").tolist()
        category_iou = segmentation_metrics.pop("per_category_iou").tolist()

        # 카테고리당(per-category) 정확도와 IoU로 메트릭 딕셔너리 업데이트
        segmentation_metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(category_accuracy)}
        )
        segmentation_metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(category_iou)}
        )

    return segmentation_metrics


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=calculate_segmentation_metrics,
)

trainer.train()

save_directory = "../../data/model/"
model.save_pretrained(save_directory)

# 모델을 추론에 사용해본다는데...코드 형식은 의미있겠지만, 어차피 같은 데이터로 하는 거라서...
load_directory = save_directory
loaded_model = SegformerForSemanticSegmentation.from_pretrained(
    load_directory, id2label=id2label, label2id=label2id
)
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor
from matplotlib import pyplot as plt

# 피처 추출기 불러오기
feature_extractor = SegformerImageProcessor()

# 입력 이미지 불러오기(아래 경로는 여러분 구글 드라이브의 알맞은 경로로 변경)
image = Image.open(
    "../../data/FoodSeg103_short/Images/img_dir/train/00000000.jpg"
).convert("RGB")

# 입력 이미지 전처리
inputs = feature_extractor(images=[image], return_tensors="pt")

# 세그먼테이션(segmentation) 예측 계산
with torch.no_grad():
    outputs = loaded_model(**inputs)
    predictions = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

# id2label 매핑을 사용한 예측된 segmentation 맵에서 흑백(회색조) 컬러 맵 생성
grayscale_map = np.zeros((predictions.shape[0], predictions.shape[1]), dtype=np.uint8)
for label_id in id2label.keys():
    grayscale_map[predictions == label_id] = label_id

# 흑백(회색조) 맵에서 PIL 이미지로 변환
segmentation_image = Image.fromarray(grayscale_map, mode="L")

# true 레이블 이미지 불러오기
true_label_path = "../../data/FoodSeg103_short/Images/ann_dir/train/00000000.png"
true_label = Image.open(true_label_path).convert("L")

# 원래 이미지, 예측된 세그멘테이션 맵 및 true 레이블 디스플레이
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image)
ax1.set_title("Original Image")
ax2.imshow(segmentation_image, cmap="gray")
ax2.set_title("Predicted Segmentation Map")
ax3.imshow(true_label, cmap="gray")
ax3.set_title("True Label")
plt.show()
