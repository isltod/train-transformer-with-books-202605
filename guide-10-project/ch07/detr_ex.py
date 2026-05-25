import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig

# Config, ImageProcessor, ObjectDetection 세 가지가 필요한 모양...
config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", config=config)
model.eval()

# 사용할 예제 이미지 열어서 전처리...
img_path = "../../data/tulip_field.png"
img = Image.open(img_path)
img = img.convert("RGB")
transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img_tensor = transform(img).unsqueeze(0)

# 훈련 없이 막바로 예측...
with torch.no_grad():
    outputs = model(img_tensor)

# 이미지 (가로, 세로)를 (세로, 가로)로 거꾸로 뒤집기...
target_sizes = torch.tensor([img.size[::-1]])
# ObjectDetection 이후에 또 무슨 프로세스인가...아무튼 이거까지 해야 예측 점수, 클래스, 경계 상자가 나오는 모양...
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)

# get_cmap("tab20")은 20개의 색상을 지닌 컬러맵으로 이미지의 개별 객체 카테고리에 고유의 색상을 배정
colors = plt.get_cmap("tab20").colors

# results["scores"]는 예측
# results["labels"]는 true 레이블
# results["boxes"]는 객체의 경계 상자
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    # 좌상단 점과 오른쪽 끝점, 아래쪽 끝점인 모양...빼줘야 넓이 높이가 되나보다...
    x, y, w, h = box
    w = w - x
    h = h - y
    rect = plt.Rectangle(
        (x, y), w, h, linewidth=1, edgecolor=colors[label % 20], facecolor="none"
    )
    ax.add_patch(rect)
    # 모델의 config.id2label에 이름이 저장된 모양...
    ax.text(x, y, f"{model.config.id2label[label.item()]}\
    {round(score.item(), 3)}", fontsize=15, color=colors[label % 20])

plt.show()
# 막상 해보니 사람과 차 정도만 잡아내고, 사람도 겹치면 못 잡는 경우도 있고, 집, 꽃 나무 등은 못 잡는 듯...
