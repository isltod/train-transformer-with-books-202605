import torch
import torch.nn as nn

# 랜덤 시드 설정
torch.manual_seed(0)

# 파라미터 정의
num_embeddings = 10  # 어휘(vocabulary) 개수
embedding_dim = 3  # 임베딩 벡터 차원

# 임베딩 층
embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
# 1번 단어, 5번 단어를 의미하는 모양이네...
input_tokens = torch.tensor([1, 5])
# 위에서 (10,3)으로 일단 임베딩 클래스 생성했고, 여기서 (1,5) 넘겨주면 __call()__ 건너 forward로 가겠지..
output_embeddings = embedding(input_tokens)
print(output_embeddings)
