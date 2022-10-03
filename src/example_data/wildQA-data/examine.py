import json
from collections import defaultdict

split = "test"
with open(f"{split}.json", 'r') as f:
    data = json.load(f)

# question_type_split = defaultdict(list)
tokens = defaultdict(int)
for d in data:
    for token in d["answer"].split():
        tokens[token] += 1
    for sent in d["alter_answers"]:
        for token in sent.split():
            tokens[token] += 1

sorted_tokens = {k: v for k, v in sorted(tokens.items(), key=lambda item: item[1])}

print(sorted_tokens)

# for t, itms in question_type_split.items():
    # print(f"{t}: {len(itms)}")
    # print(f"{t}: {sum(itms)/len(itms)}")

# for t, itms in question_type_split.items():
#     with open(f"{split}_{t}.json", 'w') as f:
#         json.dump(itms, f, indent=4)