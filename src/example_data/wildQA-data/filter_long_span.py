import json

with open("raw_test.json", 'r') as f:
    data = json.load(f)

filtred_data = []
num_filtered = 0
for d in data:
    alter_evidences = d["alter_evidences"]
    duration = d["duration"]
    filtered_aev = []
    for ev in alter_evidences:
        filtered_ev = []
        for se in ev:
            s, e = se[0], se[1]
            if e - s > 1/4 * duration:
                num_filtered += 1
                continue    
            filtered_ev.append([s, e])
        if filtered_ev:
            filtered_aev.append(filtered_ev)
    d["alter_evidences"] = filtered_aev
    filtred_data.append(d)

print(num_filtered)
with open("test.json", 'w') as f:
    json.dump(filtred_data, f, indent=4)