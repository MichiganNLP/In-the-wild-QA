#!/usr/bin/env python
import pandas as pd
import random
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the input file for the second annotation stage.")
    parser.add_argument("-bs","--batch-size",metavar="THE NUMBER OF HITS PER BATCH", 
        help='''You can control the number of hits (a task in Amazon Mechanical Turk) per batch.
        In the original paper's setting, for example, if the `batch-size` is `3`, there will be 3 
        hits (with 3*5=15 questions) in one batch''',
        type=int,default=np.inf)

    return parser.parse_args()


def select_question_inputs(df,HitNeeded,AvoidDup=True):
    ## select 5 questions for each task
    random.seed(0)
    Keys = df.columns
    OutDic = {k+f"{i}":[] for k in Keys for i in range(1,6)}
    ridxs=np.arange(len(df))
    
    for _ in range(HitNeeded):
        selected_ids = random.sample(range(len(ridxs)),5)
        selected_rows = ridxs[selected_ids]
        if AvoidDup: # avoid duplicated videos in same HIT
            trytime = 0
            while len(set(df["video"][selected_rows])) < 5:
                if trytime > 10:
                    raise RuntimeError("Hard to find unduplicated videos")
                selected_ids = random.sample(range(len(ridxs)),5)
                selected_rows = ridxs[selected_ids]
                trytime += 1
        for rid in range(5):
            crow = df.iloc[selected_rows[rid]]
            for k in Keys:
                OutDic[k+f"{rid+1}"].append(crow[k])
        ridxs = np.delete(ridxs,selected_ids)
    return OutDic


def escapeHtml(unsafe):
    return unsafe.replace("&","&amp;").replace("<","&lt;")\
        .replace(">","&gt;").replace("\"",'&quot;').replace("'","&#039;")



if __name__ == "__main__":
    BatchSize = parse_args().batch_size

    df_all_stage1_anno = pd.read_csv(
        "organize_stage1_annotation/processed_reviewed_rm_del.csv") # all questions from stage 1
    if not (pre_stg2_ipt:=os.listdir('previous_stage2_inputs')) == [".gitkeep"]:
        # if there are previous stage 2 annotation input
        df_pre_concat = []
        for pre_name in pre_stg2_ipt:
            if pre_name != ".gitkeep":
                df_pre_concat.append(pd.read_csv("previous_stage2_inputs/"+pre_name))

        df_pre_concat = pd.concat(df_pre_concat)
        used_anno = {tuple(ite) for i in range(1,6) for ite in 
        df_pre_concat[[f"video{i}",f"question{i}",f"stdAnswer{i}"]].to_dict("split")["data"]}
    else:
        used_anno = {}

    # Filter out questions that haven't been used
    feasible_anno = df_all_stage1_anno[df_all_stage1_anno.apply(
        lambda x: (x["video_link"],x['modified_question'],x["modified_answer"]) not in used_anno,axis=1)]
        
    # # This is to fix an early stage annotation issue
    # feasible_anno = feasible_anno[feasible_anno.apply(
    #     lambda x: (x["video_link"],x['modified_question'],x["correct_answer"]) not in used_anno,axis=1)]
    
    feasible_anno.reset_index(drop=True, inplace=True)

    feasible_anno = feasible_anno[['domain','modified_question','video_link','modified_answer','evidences']]
    feasible_anno = feasible_anno.rename({"modified_question":"question","video_link":"video",
                                        "modified_answer":"stdAnswer","evidences":"stdEvidences"}, axis=1)
    feasible_anno["question"] = feasible_anno["question"].apply(escapeHtml)
    # feasible_anno["stdAnswer"] = feasible_anno["stdAnswer"].apply(escapeHtml)


    print(f'\n{len(df_all_stage1_anno)} questions in total;\
        \n{len(df_all_stage1_anno) - len(feasible_anno)} questions used in previous annotation;\
        \n{len(feasible_anno)} questions left\n')

    # organize questions into Amazon Mechanical Turk input
    # each HIT has 5 questions
    new_hits = pd.DataFrame.from_dict(select_question_inputs(
        feasible_anno,HitNeeded=len(feasible_anno)//5))
    
    # divide takes into batch
    bs = min(len(new_hits),BatchSize)
    for i in range((len(new_hits)+bs-1)//bs):
        new_hits.iloc[i*bs:min((i+1)*bs,len(new_hits))]\
            .to_csv(f"second_stage_annotation_input_{i}.csv",index=False)


    # In the original paper, we seperate the `military` domain from 
    # the others, and the following codes are account for this setting
    '''
    mil_feasible_anno = feasible_anno[feasible_anno["domain"]=="Military"]
    no_mil_feasible_anno = feasible_anno[feasible_anno["domain"]!="Military"]
    mil_feasible_anno.reset_index(drop=True, inplace=True)
    no_mil_feasible_anno.reset_index(drop=True, inplace=True)

    print(f'\n{len(df_all_stage1_anno)} questions in total;\
        \n{len(df_all_stage1_anno) - len(feasible_anno)} questions used in previous annotation;\
        \n{len(feasible_anno)} questions left\n')
    print("{} Mil questions,{} no-Mil;{} in all".
        format(len(mil_feasible_anno),
                len(no_mil_feasible_anno),
                len(feasible_anno)))

    # organize questions to Amazon Mechanical Turk input
    # each task has 5 questions
    mil_hits = pd.DataFrame.from_dict(select_question_inputs(
        mil_feasible_anno,HitNeeded=len(mil_feasible_anno)//5))
    no_mil_hits = pd.DataFrame.from_dict(select_question_inputs(
        no_mil_feasible_anno,HitNeeded=len(no_mil_feasible_anno)//5))
    
    # divide takes into batch
    MilBatchSize = min(len(mil_hits),BatchSize)
    for i in range((len(mil_hits)+MilBatchSize-1)//MilBatchSize):
        mil_hits.iloc[i*MilBatchSize:min((i+1)*MilBatchSize,len(mil_hits))]\
            .to_csv(f"mil_stage2_input_{i}.csv",index=False)
    NoMilBatchSize = min(len(no_mil_hits),BatchSize)
    for i in range((len(no_mil_hits)+NoMilBatchSize-1)//NoMilBatchSize):
        no_mil_hits.iloc[i*NoMilBatchSize:min((i+1)*NoMilBatchSize,len(no_mil_hits))]\
            .to_csv(f"no_mil_stage2_input_{i}.csv",index=False)
    '''
