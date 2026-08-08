import torch
from gen_text_simple import tokenizer, GPT_CONFIG_124M, model
from previous_chapters import create_dataloader_v1

# 1. 데이터를 로드하고
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
# 처음 99개 문자
print(text_data[:99])
# 마지막 99개 문자
print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("문자:", total_characters)
print("토큰:", total_tokens)


# 2. 데이터 로더를 준비한다
# 훈련 세트 비율
train_ratio = 0.90
# 진짜 이런 식으로 토큰화하지 않은 상태에서 먼저 문장을 나누네...글자 단위로 엉뚱한 데서 끊어지는데...
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
print(train_data[-10:])
print(val_data[:10])

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    # 검증 데이터는 마지막에 배치 크기 안 맞아도 사용하네...
    drop_last=False,
    shuffle=False,
    num_workers=0,
)
# 데이터 로더 테스트...
print("훈련 데이터 로더:")
for x, y in train_loader:
    # 배치 2이고 예제는 문장 길이를 256개 단어로 한다고 했으니 (2, 256) 나올테고...
    print(x.shape, y.shape)

print("\n검증 데이터 로더:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
# 데이터 로더에 enumerate 같은 별다른 처리 안해줘도 for 반복마다 입력과 정답지 배치를 반환하는 모양...
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("훈련 토큰 수:", train_tokens)
print("검증 토큰 수:", val_tokens)
print("모든 토큰 수:", train_tokens + val_tokens)


def calc_loss_batch(input_batch, target_batch, model, device):
    # gpu로 배치 데이터 옮기고
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 순전파로 로짓(소프트맥스가 아니다...) 예측하고
    logits = model(input_batch)
    # 펼친 로짓과 펼친 정답지 ID로 크로스 엔트로피
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    # 데이터 로더에 데이터 없으면 nan 반환하고 종료...
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # 배치 수 없으면 한번에 배치 하나씩 그냥 모든 데이터를 처리...
        num_batches = len(data_loader)
    else:
        # num_batches가 데이터 로더에 있는 배치 개수보다 크면
        # num_batches를 데이터 로더에 있는 총 배치 개수로 맞춥니다.
        num_batches = min(num_batches, len(data_loader))

    print("배치 수:", num_batches)
    # 그러니까, 원래 데이터로더만 돌려도 배치들이 반환되는데, enumerate를 하면 앞에 인덱스를 붙여준다...
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 배치 수 이내에서...
        if i < num_batches:
            # 배치의 크로스 엔트로피 손실을 구하고 누적...
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 평균 크로스 엔트로피 손실을 반환
    return total_loss / num_batches


# 장치 설정인데 윈도우에서는 mps가 필요없는거 아닌가?
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # 파이토치 2.9 이상에서는 mps 결과가 안정적입니다.
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
print(f"Using {device} device.")

# nn.Module 클래스의 경우 model = model.to(device)로 할당할 필요가 없습니다.
model.to(device)

# 데이터 로더에서 셔플링이 일어나므로 재현가능성을 위해 설정합니다.
torch.manual_seed(123)

with torch.no_grad():  # 모델을 아직 훈련하지 않으므로 효율성을 위해 그레이디언트 추적을 끕니다.
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("훈련 손실:", train_loss)
print("검증 손실:", val_loss)
