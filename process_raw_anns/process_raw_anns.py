import json
import os

DOMAINS = ["Geography", "Military"]
MAX_VID_IDX = 5
MAX_QUESTION_NUM = 100
MAX_BAR_NUM = 5


def process(data):
    worker_id_idx = data[0].split(",").index("\"WorkerId\"")
    title_idx = data[0].split(",").index("\"Title\"")
    time_spent_idx = data[0].split(",").index("\"WorkTimeInSeconds\"")
    vids_idxs = {i: data[0].split(",").index(f"\"Input.video{i}\"") for i in range(1, 6)}

    data = data[1:]
    
    questions = []
    for d in data:
        # load the annotation json string
        idx = d.find("\"[{\"\"")
        text = d[idx:].replace("\"\"", "\"")
        if text.endswith("\n"):
            text = text[1:-2]   # delete the starting and ending "  
        else:
            text = text[1:-1]
        ann = json.loads(text)     
        assert len(ann) == 1

        ann = ann[0]

        # clean the annotations
        for k, itm in ann.items():
            if k.startswith("confidence"):
                confidence_level = [level for level, bln in itm.items() if bln]
                assert len(confidence_level) == 1
                ann[k] = confidence_level[0]
        
        # Add other annotator informations
        info_toks = d.replace(", ", " ")
        info_toks = info_toks.split(",")
        domain = next((dm for dm in DOMAINS if dm in info_toks[title_idx]), "")
        assert domain

        worker_id = info_toks[worker_id_idx][1:-1]
        time_spent = info_toks[time_spent_idx][1:-1]
        vid_links = {vid_id: info_toks[pos][1: -1] for vid_id, pos in vids_idxs.items()}

        # video_ids is for mapping of the visual features
        video_ids = {vid_id: info_toks[pos][1: -1].split(".mp4")[0].split("/")[-1] for vid_id, pos in vids_idxs.items()}

        assert all([link.startswith("https://www.dropbox.com/s/") for _, link in vid_links.items()])

        # convert to question-answer-wise format
        vid_idx = 1
        while vid_idx <= MAX_VID_IDX:
            objective = ann[f"objective-input-{vid_idx}"]
            q_idx = 0

            while q_idx < MAX_QUESTION_NUM:
                if f"confidence-{vid_idx}-{q_idx}" not in ann:
                    break
                confidence = ann[f"confidence-{vid_idx}-{q_idx}"]
                correct_ans = ann[f"correct-answer-{vid_idx}-{q_idx}"]

                default_types = [tp for tp, bln in ann[f"how-{vid_idx}-{q_idx}"].items() if bln]
                if not default_types:
                    assert ann[f"other-{vid_idx}-{q_idx}"]
                    default_types = [ann[f"other-{vid_idx}-{q_idx}"]]

                question_type = default_types

                # multiple bases could be possible
                question_base = [tp for tp, bln in ann[f"question-based-{vid_idx}-{q_idx}"].items() if bln]

                question = ann[f"question-input-{vid_idx}-{q_idx}"]

                bar_idx = 0
                evidences = []
                while bar_idx < MAX_BAR_NUM:
                    if f"start-bar-{vid_idx}-{q_idx}-{bar_idx}" not in ann:
                        break
                    evidences.append({
                        bar_idx: [ann[f"start-bar-{vid_idx}-{q_idx}-{bar_idx}"], ann[f"end-bar-{vid_idx}-{q_idx}-{bar_idx}"]]
                    })

                    bar_idx += 1
                q_idx += 1
                questions.append({
                    "objective": objective,
                    "confidence": confidence,
                    "question": question,
                    "correct_answer": correct_ans,
                    "question_type": question_type,
                    "question_base": question_base,
                    "evidences": evidences,
                    "domain": domain,
                    "worker_id": worker_id,
                    "time_spent": time_spent,
                    "video_link": vid_links[vid_idx],
                    "video_id": video_ids[vid_idx]
                })
            vid_idx += 1
    return questions


if __name__ == "__main__":
    raw_files = os.listdir("raw_batches")
    questions = []
    for fn in [raw_file for raw_file in raw_files if raw_file.endswith(".csv")]:
        with open(f"raw_batches/{fn}", 'r') as f:
            data = f.readlines()

        questions.extend(process(data))

    with open("processed_result/annotations.json", 'w') as f:
        json.dump(questions, f, indent=4)
