from tokenizers import Tokenizer, models, pre_tokenizers, trainers

"""
1. 학습 과정인데, 데이터셋 준비하고, BPE 토크나이저 만들고, 전처리 토크나이저 달고, Trainer 달고, 학습...
"""
# 데이터셋 불러오기
with open("../../data/tokenizer_train.txt", "r") as file:
    dataset = [line.strip() for line in file.readlines()]

# BPE tokenizer 인스턴스화
tokenizer = Tokenizer(models.BPE())

# 입력을 words로 바꾸기 위해 pre-tokenizer 설정
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# BPE tokenizer를 데이터셋으로 학습시킴
trainer = trainers.BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train_from_iterator(dataset, trainer=trainer)

tokenizer.save("../../data/tokenizer.json")

"""
2. 추론 과정으로, 저장한 json 토크나이저 불러오고, encode 해서 출력해본다...
"""
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="../../data/tokenizer.json")
text = "The Tokenizers"
encoded = tokenizer.encode(text)

# 토큰화된 텍스트 출력
print(encoded.tokens)

# # 토큰화 프로세스 시각화 - 이건 별것도 없고 스크립트에서는 실행도 안된다...
# from tokenizers.tools import EncodingVisualizer
# visualizer = EncodingVisualizer(fast_tokenizer._tokenizer)
# visualizer(text="The Tokenizers")
