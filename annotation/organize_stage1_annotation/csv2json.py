import json
import pandas as pd


if __name__ =="__main__":
    df = pd.read_csv("./processed_reviewed_rm_del.csv",
        converters={"question_type":eval,"question_base":eval,
        "evidences":eval,"evidences_in_min":eval,
        "time_in_original_video":eval})
    questions = df.to_dict(orient="records")
    
    with open("./processed_reviewed_rm_del.json", 'w') as f:
        json.dump(questions, f, indent=4)
