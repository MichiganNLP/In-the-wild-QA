import json
import pandas as pd


if __name__ =="__main__":
    df = pd.read_csv("./merged_annotation.csv",
        converters={"question_type":eval,"question_base":eval,
        "evidences":eval,"evidences_in_min":eval,
        "time_in_original_video":eval,
        "crowd_answers": eval,
        "crowd_evidences": eval,
        "crowd_deleted_evidences": eval,
        "crowd_answer_based_on": eval,
        "crowd_confidence": eval,
        "crowd_workerids": eval,
        "expert_answers": eval,
        "expert_evidences": eval,
        "expert_deleted_evidences": eval,
        "expert_answer_based_on": eval,
        "expert_confidence": eval,
        "expert_workerids":eval
        })
    questions = df.to_dict(orient="records")
    
    with open("./merged_annotation.json", 'w') as f:
        json.dump(questions, f, indent=4)
