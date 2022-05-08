from xxlimited import new
import pandas as pd
import re
import json
from collections import defaultdict
import os


def order_worker_answers_evidences_by_question(answer, durations, count=5):
  start_bar_keys = [k for k in answer.keys() if k.startswith("start-bar-")]
  return [[[answer[f"answer-input-{i+1}-{subid}"], # answer
            [[float(answer[f"start-bar-{i+1}-{subid}-{evid}"])/100.0*durations[i],
              float(answer[f"end-bar-{i+1}-{subid}-{evid}"])/100.0*durations[i]]
             for evid in sorted(set([int(k.split("-")[4]) for k in start_bar_keys
             if k.startswith(f"start-bar-{i+1}-{subid}")]))], # evidence
             [tp for tp, bln in answer[f"answer-based-{i+1}-{subid}"].items() if bln], # answer-vased
             [tp for tp, bln in answer[f"confidence-{i+1}-{subid}"].items() if bln][0] # confidence
             ]
          for subid in sorted(set([int(key.split("-")[3]) for key in start_bar_keys 
          if key.startswith(f"start-bar-{i+1}")]))] for i in range(count)]


def replace_special_video_id(vid:str):
    return vid.replace("Olly_s","Olly's")          


def parse_hits(df, include_reject=False):
  def _minsec2sec(t):
    if ":" not in t:
      return float(t)
    else:                
      mins, sec = t.split(":")
      return float(mins) * 60 + float(sec)

  if not include_reject:
    df = df.loc[df["RejectionTime"].isnull()]
  hits = df.groupby(["HITId"] + 
                      [c for c in df.columns if c.startswith("Input.")])\
    .agg({"Answer.taskAnswers": lambda lists: [x for list_ in lists for x in list_],
          "AssignmentId": list,"WorkerId": list, "ApprovalTime": list}).\
          reset_index().to_dict("records") # attention: groupby will also sort the keys
  
  for hit in hits:
    durations = [general_info[replace_special_video_id(hit[f"Input.video{vid}"]).split(".mp4")[0].split("/")[-1]]["seconds"] for vid in range(1,6)]
    durations = [ _minsec2sec(x["end"])-_minsec2sec(x["start"]) for x in durations]
    hit["answers_evidences_by_worker"] = {worker_id: {**{f"question{i + 1}": q_answers
                                                    for i, q_answers in
                                                    enumerate(order_worker_answers_evidences_by_question(worker_answers,durations))},
                                                  **{"comments": worker_answers.get("comments")}}
                                      for worker_id, worker_answers in zip(hit["WorkerId"], hit["Answer.taskAnswers"])}
    del hit["Answer.taskAnswers"]
    del hit["AssignmentId"]
    del hit["WorkerId"]
    del hit["ApprovalTime"]
  return {hit["HITId"]: hit for hit in hits}


if __name__ == "__main__":
    with open("general_info.json", 'r') as f:
        general_info = json.load(f)
    general_info = {itm["video_name"].split(".mp4")[0]: itm["time_in_original_video"] for itm in general_info}

    for annotator in ["expert","crowd"]:
        raw_files = os.listdir(f"{annotator}_raw_batches")
        csv_files = []
        for fn in [raw_file for raw_file in raw_files if raw_file.endswith(".csv")]:
            new_df = pd.read_csv(f"{annotator}_raw_batches/{fn}",converters={"Answer.taskAnswers": json.loads})
            # fix some early annotation stage issue
            if "Input.StdAnswer1" in new_df.columns:
              new_df.rename({f"Input.StdAnswer{i}":f"Input.stdAnswer{i}" for i in range(1,6)},
                inplace=True,axis=1)
            if "Input.StdEviden1" in new_df.columns:
              new_df.rename({f"Input.StdEviden{i}":f"Input.stdEvidences{i}" for i in range(1,6)},
                inplace=True,axis=1)
            for kk in range(1,6):
              new_df[f"Input.file{kk}"] = fn
            csv_files.append(new_df)
        if len(csv_files) == 0:
            continue
        anno_result = pd.concat(csv_files,join="inner")
        hits = parse_hits(anno_result)
        # write out csv
        anno_df = pd.DataFrame.from_dict(
            {"file":[hit[f'Input.file{i+1}'] for hit in hits.values() for i in range(5)],
            "domain":[hit[f'Input.domain{i+1}'] for hit in hits.values() for i in range(5)],
            "question":[hit[f'Input.question{i+1}'] for hit in hits.values() for i in range(5)],
            "video":[hit[f'Input.video{i+1}'] for hit in hits.values() for i in range(5)],
            "stdAnswer":[hit[f'Input.stdAnswer{i+1}'] for hit in hits.values() for i in range(5)],
            "stdEvidences":[hit[f'Input.stdEvidences{i+1}'] for hit in hits.values() for i in range(5)],
            "answers":[[[ans[0] for ans in wok[f"question{i+1}"]] 
                        for wok in list(hit['answers_evidences_by_worker'].values()) ] for hit in hits.values() for i in range(5)],
            "evidences":[[[ans[1] for ans in wok[f"question{i+1}"]] 
                        for wok in list(hit['answers_evidences_by_worker'].values()) ] for hit in hits.values() for i in range(5)],
            "answer_based_on":[[[ans[2] for ans in wok[f"question{i+1}"]] 
                        for wok in list(hit['answers_evidences_by_worker'].values()) ] for hit in hits.values() for i in range(5)],
            "confidence":[[[ans[3] for ans in wok[f"question{i+1}"]] 
                        for wok in list(hit['answers_evidences_by_worker'].values()) ] for hit in hits.values() for i in range(5)],
            "workerids":[list(hit['answers_evidences_by_worker'].keys()) for hit in hits.values() for _ in range(5)]
            })

        anno_df.to_csv(f"organized_{annotator}_ans.csv",index=False)
