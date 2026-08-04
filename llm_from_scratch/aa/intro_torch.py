import numpy as np
import torch

"""
1. 버전, GPU 확인
"""
print(torch.__version__)
print(torch.cuda.is_available())

"""
2. 텐서 랭크, 값 복사와 참조
"""
# 그냥 숫자를 쓰면 0D tensor (scalar)
tensor0d = torch.tensor(1)

# 대괄호 하나면 1D tensor (vector)
tensor1d = torch.tensor([1, 2, 3])

# 대괄호 2이면 2D tensor
tensor2d = torch.tensor([[1, 2], [3, 4]])

# 셋이면 3D tensor
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 또는 넘파이 배열에서...
ary3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tensor3d_val = torch.tensor(ary3d)  # 값을 복사해서 새로 만들기
tensor3d_ref = torch.from_numpy(ary3d)  # 넘파이 배열과 메모리를 공유?

# 확인을 위해서 원본의 맨 처음 요소 값을 바꿔보면...
ary3d[0, 0, 0] = 999
print(tensor3d_val)  # 값을 복사한 경우는 상관없고
print(tensor3d_ref)  # 참조한 경우는 값이 바뀌어있고...

"""
3. 데이터 타입
"""
# 정수는 기본적으로 int64
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)
# 실수는 기본적으로 float32
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
# int64 -> float32 변환
floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)
