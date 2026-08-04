import numpy as np
import torch

"""
A.1. 버전, GPU 확인
"""
print(torch.__version__)
print(torch.cuda.is_available())

"""
A.2.1 텐서 랭크, 값 복사와 참조
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
A.2.2 데이터 타입
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

"""
A.2.3 일반적인 연산자
"""
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d)

# 같은 결과 size(2,3) 출력
print(tensor2d.shape)
print(tensor2d.size())

# 결과는 같지만, view는 그냥 보기만 바꾸니 값들이 붙어있어야 하고,
# reshape는 복사해서 다시 만들어 값이 붙어있지 않아도 된다고...
aa = tensor2d.view(3, 2)
bb = tensor2d.reshape(3, 2)
print(aa)
print(bb)
# 근데 원본을 바꾸면 둘 다 바뀌어있기는 마찬가지다...
tensor2d[0, 0] = 999
print(aa)
print(bb)

print(tensor2d.T)
# 행렬곱은 메서드 matmul 또는 @ 연산자
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)

"""
A.3 계산그래프
"""
import torch.nn.functional as F

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1])  # input feature
# 학습 매개변수는 뒤에서 grad() 적용시키려면 requires_grad=True 있어야 하는모양...
w1 = torch.tensor([2.2], requires_grad=True)  # weight parameter
b = torch.tensor([0.0], requires_grad=True)  # bias unit

z = x1 * w1 + b  # net input
a = torch.sigmoid(z)  # activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)
# 여기까지 순전파, 계산그래프를 만든다...
# 여기부터 역전파
from torch.autograd import grad

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)
print(grad_L_w1)
print(grad_L_b)

# 또는 더 고수준 backward() 함수 사용...근데 위에서 grad 하고 다시 backward 했는데 .grad 속성이 누적되지 않네?
loss.backward()
print(w1.grad)
print(b.grad)
