import torch
import torch.nn as nn


# 클래스로 캡슐화
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 분모가 0이 안되게...
        self.eps = 1e-5
        # 정규화 결과를 늘리고 이동시키는 매개변수들인데, 모델 성능에 도움이 된다면 학습해서 변형시킨다...
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 정규화는 마지막 차원에 적용...
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False 또는 토치 2.0이상에서 correction=0은 데이터 수 N으로 나누는 모분산...반대는 N-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 분모가 0이 안되게...
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)

    # 다섯 개의 차원(특성)을 가진 두 개의 훈련 샘플을 만듭니다.
    batch_example = torch.randn(2, 5)

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    # 원래 마지막 차원이 쪼그러들어서 (1,2)로 나올걸 원래와 비슷한 차원을 유지해서 (2,1)로 나오게 keepdim
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    # 당연히 결과는 평균 0, 분산 1이 아니다..
    print("평균:\n", mean)
    print("분산:\n", var)

    # 평균 빼고 분산 제곱근으로 나누면 정규화
    out_norm = (out - mean) / torch.sqrt(var)
    print("정규화된 층 출력:\n", out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("평균:\n", mean)
    print("분산:\n", var)

    # 클래스를 사용해서 캡슐화
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("평균:\n", mean)
    print("분산:\n", var)

    # 배치 정규화는 배치 차원을 따라서, 층 정규화는 특성 차원을 따라서 정규화...
    # 배치 크기 변화를 걱정할 필요가 없고 분산 훈련이나 작은 환경에 배포할 때 유리하다고...
