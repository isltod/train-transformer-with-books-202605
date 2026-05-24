import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

img_path = "../../data/tulip_field.png"

# 이미지 열기
img = Image.open(img_path)

# RGB로 변환
img = img.convert("RGB")
img.show()

# 데이터 증강 파이프라인 정의
transforms = T.Compose(
    [
        T.RandomRotation(degrees=(-15, 15), fill=0),
        # 스케일은 이미지 크기를 80~100% 무작위 조절
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 이미지에 데이터 증강 파이프라인을 적용
augmented_img = transforms(img)

# 이미지를 시각화하기 위해 해당 이미지를 PIL 이미지로 변환
# 다만, 이 변환 이전에 정규화(normalization) 취소 조치 실행
unnormalized_img = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
        T.ToPILImage(),
    ]
)(augmented_img)

unnormalized_img.show()
