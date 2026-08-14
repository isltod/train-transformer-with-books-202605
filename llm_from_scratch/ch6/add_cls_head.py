import torch
from init_model import model, BASE_CONFIG
from make_datatloader import tokenizer

# 모델 동결 - 결국 가중치들이 학습되지 않도록 설정하는 거긴 한데...전에 이거 말고 뭔가 더 있던 같긴 한데...
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
# 그리고는 출력 층을 2가지로 분류하는 헤드로 교체...이러면 새로 넣은 마지막 층은 자동으로 requires_grad = True 라고...
num_classes = 2
model.out_head = torch.nn.Linear(
    # 히든 = 임베딩 차원을 2로 축소...
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes,
)

# 트랜스포머 블록의 마지막 트랜스포머 층들은 학습
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

# 그리고 마지막 정규화 층도 학습
for param in model.final_norm.parameters():
    param.requires_grad = True
# 이렇게 세 가지를 추가로 학습시키면 결과가 좋아진다고...나중에 나도 이런 실험을 해봐야겠다...

if __name__ == "__main__":
    # 바꾸기 전에 모델 구조 출력
    print(model)

    # 일단 이 모델이 작동하는 방식을 테스트해보면...우선 스팸검사 텍스트를 넣고...
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("입력:", inputs)
    # 이 예제의 입력 모양은 (배치 1, 토큰 ID 4)가 되고...
    print("입력 차원:", inputs.shape)
    # 결과를 받는데...
    with torch.no_grad():
        outputs = model(inputs)
    print("출력:\n", outputs)
    # 출력은 (배치 크기 1, 토큰 ID 4, 클래스 수 2)가 나온다...
    print("출력 차원:", outputs.shape)
    # 근데 이 모델의 작동 방식을 잘 생각해보면
    # 앞에 3개 토큰은 입력의 2~4 단어를 반복한 것 뿐이고, 실제 모델 예측은 마지막 단어 뿐이다...
    # 따라서 스팸인지 아닌지에 대한 학습 결과는 마지막 단어에 출력된다고 생각하는 것이 맞다...
    print("마지막 출력 토큰:", outputs[:, -1, :])
