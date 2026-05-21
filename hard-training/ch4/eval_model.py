from pprint import pprint

import evaluate

# 이걸로 미리 만들어진 평가지표(그 중 정확도)를 불러온다...
# 근데 대신에 다운로드한다고 처음에 시간이 좀 걸리는 거 같다...
acc = evaluate.load("accuracy")
# 여러개를 조합해서 사용할 수도 있다고...
metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
metrics.compute(predictions=[1, 0, 0, 1], references=[0, 1, 0, 1])
print(metrics)

for y, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
    # add로 더하기만 하고
    metrics.add(predictions=pred, references=y)
# 한 번에 compute할 수도...
pprint(metrics.compute())

for y, preds in zip([[0, 1], [0, 1]], [[1, 0], [0, 1]]):
    # add와 비슷한데 두 개씩 묶어서 배치로 처리하는 경우...
    metrics.add_batch(predictions=preds, references=y)
pprint(metrics.compute())
