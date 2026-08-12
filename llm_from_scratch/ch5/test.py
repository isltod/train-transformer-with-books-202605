import os
import torch

# 1. GPU 사용 가능 여부 확인
print("GPU 사용 가능:", torch.cuda.is_available())

# 2. 사용 가능한 GPU 개수 출력
device_count = torch.cuda.device_count()
print("GPU 개수:", device_count)

# 3. 각 GPU의 이름 및 상세 정보 출력
for i in range(device_count):
    print(f"--- GPU [{i}] ---")
    print("이름:", torch.cuda.get_device_name(i))
    print("메모리 할당량:", torch.cuda.memory_allocated(i))
    print("기기 속성:", torch.cuda.get_device_properties(i))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("현재 GPU 번호:", torch.cuda.current_device())
    print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
elif torch.backends.mps.is_available():
    # 파이토치 2.9 이상에서는 mps 결과가 안정적입니다.
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")
    device = torch.device("cpu")
