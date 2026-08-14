from add_cls_head import model
from make_datatloader import train_loader, val_loader, test_loader
import torch


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    # 배치 수가 없다면 그냥 데이터 크기 그대로..배치는 1
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 아니면 배치 수가 데이터 수보다 많은지 체크해서 정리...
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        # 주어진 배치 수 내에서...
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # 마지막 출력 토큰의 로짓만 본다는 것에 주의...
                logits = model(input_batch)[:, -1, :]
            # 소프트맥스 없이 바로 argmax로 최대 확률 찾고
            predicted_labels = torch.argmax(logits, dim=-1)

            # 배치 수만큼 확인한 데이터 수 누적하고
            num_examples += predicted_labels.shape[0]
            # 맞는 대답은 예측과 정답지 일지하는 경우...배치는 sum
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            # 주어진 배치 수만큼 검사했으면 나가기
            break
    # 정확도는 (맞는 대답 / 확인한 데이터 수)
    return correct_predictions / num_examples


# 전과 거의 같지만 마지막 토큰만 본다는 점이 다르다...
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 여기가 다른 부분...마지막 출력 토큰의 로짓만 본다...
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# 이건 전과 똑같은데, 위에서 바꾼 calc_loss_batch를 불러써야 하니 다시 선언한다...
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


if __name__ == "__main__":
    # 이건 뭔가 from 문에서 꼬였나보다...gpu가 안잡힌다...그래도 멈추지는 않으니 우선 그냥 사용...
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
    print(f"실행 장치: {device}")

    # nn.Module 클래스의 경우 model = model.to(device) 할당문이 필요하지 않습니다. -> 그래서 그냥 model.to(device)
    model.to(device)

    torch.manual_seed(123)

    # 아직 미세 조정 없으니 정확도 자체는 의미는 없지만...
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(f"훈련 정확도: {train_accuracy*100:.2f}%")
    print(f"검증 정확도: {val_accuracy*100:.2f}%")
    print(f"테스트 정확도: {test_accuracy*100:.2f}%")

    # 훈련하는 게 아니니 역전파 끄고 훈련/검증/시험 손실을 계산하는데...숫자는 의미없고...
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(f"훈련 손실: {train_loss:.3f}")
    print(f"검증 손실: {val_loss:.3f}")
    print(f"테스트 손실: {test_loss:.3f}")
