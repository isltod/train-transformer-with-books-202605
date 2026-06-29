from importlib.metadata import version

print("파이토치 버전:", version("torch"))
print("tiktoken 버전:", version("tiktoken"))
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

print(tokenizer.special_tokens_set)
print(tokenizer.encode(text, allowed_special="all"))

strings = tokenizer.decode(integers)
print(strings)

# 소설 읽어서 토크나이저로 인코딩하고
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# 좀 더 흥미로운 구절을 만들기 위해 처음 50개 삭제? 뭔 얘긴지...
enc_sample = enc_text[50:]

# 윈도우 4로 문제와 정답지? 만들어?
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

# 문제와 정답지 출력?
for i in range(1, context_size + 1):
    # i 바로 전까지가 문제, i가 정답?
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    # tiktoken은 decode 할 때 리스트를 받아야 하는 모양이지?
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
