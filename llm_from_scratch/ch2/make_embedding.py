import torch

# 입력 단어는 4개 있고
input_ids = torch.tensor([2, 3, 5, 1])

# 총 단어수는 6, 단어 표현은 3차원이라면
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
# 임베딩 레이어를 만들 때 총 단어 수, 표현 차원 이렇게 넣는데...
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# 만들어진 임베딩은 (6,3)이라고...
print(embedding_layer.weight)

# 두 번째 단어 아이디 3을 넣으면...4번째 행의 벡터가 나오는데...
print(embedding_layer(torch.tensor([3])))
# 아이디들을 넣으면...해당 행의 벡터가 순서대로...
# 가중치 볼 때는 weight 속성을 썼는데 벡터를 뽑을 때는 weight 속성을 안쓰고 그냥 인덱싱으로 바로...
print(embedding_layer(input_ids))
