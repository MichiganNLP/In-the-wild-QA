import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("to_review.csv")

    df = df[["domain","assignment_id","video_link","question","modified_question","correct_answer","modified_answer"]]\
        .dropna(how="all") # remove empty records
    df["modified_question"] = df["modified_question"].fillna(df["question"]) # complement to whole questions
    df["modified_answer"] = df["modified_answer"].fillna(df["correct_answer"])
    df[["domain","assignment_id","video_link","question"]] = df[["domain","assignment_id","video_link","question"]]\
        .fillna(method="ffill") # fillin default values for new-adding modified questions
    df.to_csv("reviewed.csv",index=False)
