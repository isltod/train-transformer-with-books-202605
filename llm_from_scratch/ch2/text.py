# from importlib.metadata import version
#
# print("파이토치 버전:", version("torch"))
# print("tiktoken 버전:", version("tiktoken"))

# 1단계: 텍스트 읽고
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("총 문자 개수:", len(raw_text))
print(raw_text[:99])

import re

# 2단계: 토큰으로 쪼개기
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

# 3단계: 토큰을 토큰ID로 바꾸기
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# 다 모아서 클래스를 만들었다는데...vocab은 위에 거 다 해야 나오는건데, 결국 위 코드가 다 필요하단 얘긴데...
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 'hello,. world'

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 구둣점 문자 앞의 공백을 삭제합니다.
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
print(tokenizer.decode(tokenizer.encode(text)))

# 4단계: 토큰ID를 임베딩 벡터로 변환
