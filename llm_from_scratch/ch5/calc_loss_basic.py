import torch
from gen_text_simple import model, token_ids_to_text, tokenizer, GPT_CONFIG_124M

# 1. 간단한 샘플 텍스트에 대해서 다음 단어를 예측하면...
# ["every effort moves", "I really like"]
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
# ["effort moves you", "really like chocolate"]
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)

# 어휘 사전의 각 토큰에 대한 확률 (batch_size, num_tokens, vocab_size)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)
# 최대 확률로 선택된 단어 ID들...
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("토큰 ID:\n", token_ids)
print(f"첫 번째 샘플의 타깃: {token_ids_to_text(targets[0], tokenizer)}")
print(f"첫 번째 샘플의 예측: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# 당연히 학습이 안되어 있으니 엉망진창...

# 2. 정답지의 소프트맥스 확률을 구하고...
# 먼저 2개 정답 텍스트에 대한 확률을 보면...대략 10^-5~-6 오더 정도...1/50257이므로 당연...
text_idx = 0
# 첫 번째 배치, 단어 0, 1, 2 셋에, 정답지 ID 3026, 6100, 345번 확률은?
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]  # [2, 3, 50257]
print("텍스트 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("텍스트 2:", target_probas_2)

# 3. 그거의 로그 평균을 구하고
# 토큰 확률의 로그를 계산합니다.
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# 각 토큰에 대한 평균 확률을 계산합니다.
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# 지금은 평균이 -10 이하의 음수로 큰 값인데, 이걸 -를 취해 양수로 만든다..
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# 이 값을 0에 가깝게 만드는 것이 목표...
# 이 값을 크로스 엔트로피라고...torch.nn.functional.cross_entropy 함수를 이용하는데...

# 얘는 위 데이터를 그냥 쓰는게 아니고 펼쳐서 넣어줘야 한다고...
# 로짓의 크기는 (batch_size, num_tokens, vocab_size)입니다.
print("로짓 크기:", logits.shape)
# 타깃의 크기는 (batch_size, num_tokens)입니다.
print("타깃 크기:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# 얘는 (6,20257)로 20257에 각 단어들에 대한 확률이 들어있고...
print("펼친 로짓:", logits_flat.shape)
# 얘는 (6,)으로 각 단어들 ID가 있으니, 이 ID로 위에 확률 차원의 인데스로 찾으면 되는 구조...
print("펼친 타깃:", targets_flat.shape)

# 펼친 데이터를 토치의 크로스 엔트로 함수를 사용하면 이 과정을 처리해 주는 거라고...
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# 그걸 다시 exp 해주면 혼잡도라는데...나온 숫자 48,725개 단어들 중에서 뭘 선택할지 불확실하다는 의미가 된다고...
perplexity = torch.exp(loss)
print(perplexity)
