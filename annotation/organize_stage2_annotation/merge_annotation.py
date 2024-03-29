# coding: utf-8

import pandas as pd
import os
import argparse


def look_dic(x,anno_dic,anno_dic_tmp,annotator):
    if (k:=tuple(x[["video_link","modified_question","modified_answer"]])) in anno_dic.index:
        rec = anno_dic.loc[k]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])
            
    # the following codes are to fix early stage annotation issue
    elif (k:=tuple(x[["video_link","modified_question","correct_answer"]])) in anno_dic.index:
        rec = anno_dic.loc[k]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])
    elif (k:=tuple(x[["video_link","question","modified_answer"]])) in anno_dic.index:
        rec = anno_dic.loc[k]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])
    elif (k:=tuple(x[["video_link","question","correct_answer"]])) in anno_dic.index:
        rec = anno_dic.loc[k]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])
    elif (k:=tuple(x[["video_link","modified_question"]])) in anno_dic_tmp.index:
        rec = anno_dic_tmp.loc[k].iloc[0]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])
    elif (k:=tuple(x[["video_link","question"]])) in anno_dic_tmp.index:
        rec = anno_dic_tmp.loc[k].iloc[0]
        for cn in col_names:
            x[f"{annotator}_{cn}"].extend(rec[cn])


def parse_args():
    parser = argparse.ArgumentParser(description="Merge the stage 1 and stage 2 annotations.")
    parser.add_argument('-qf','--question_file',type=str,metavar="QUESTION_FILE",
        help="Path to the organized stage 1 annotation rusult (csv) file.",
        default="../organize_stage1_annotation/processed_reviewed_rm_del.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    col_names = ['answers','evidences','deleted_evidences','answer_based_on', 'confidence', 'workerids']
    question_df = pd.read_csv(args.question_file)

    for annotator in ["crowd","expert"]:
        if os.path.exists(f"organized_{annotator}_ans.csv"):
            anno_df = pd.read_csv(f"organized_{annotator}_ans.csv",converters={cn:eval for cn in col_names})
            anno_df_tmp = anno_df.copy(deep=True) # fix early stage annotation issue
            anno_df.set_index(["video","question","stdAnswer"],inplace=True)
                # TODO: cope with the repeat index situation. strategy: merge records of the same index
            anno_df_tmp.set_index(["video","question"],inplace=True)
            anno_df.sort_index(inplace=True)
            anno_df_tmp.sort_index(inplace=True)
            for cn in col_names:
                if not f"{annotator}_{cn}" in question_df:
                    question_df[f"{annotator}_{cn}"] = [[] for _ in range(len(question_df))]
            question_df.apply(look_dic,args=(anno_df,anno_df_tmp,annotator,),axis=1)

    question_df.to_csv("merged_annotation.csv",index=False)
