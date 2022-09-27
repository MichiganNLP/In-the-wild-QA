# Usage
In the original paper setting, we employed <u>both</u> "experts" and "crowdworkers" at the second stage annotation. Put your Amazon Mechanical Turk annotation files into [`expert_raw_batches`](expert_raw_batches/) and [`crowd_raw_batches`](crowd_raw_batches/) accordingly. If your annotation doesn't differentiate between "experts" and "crowdworkers", you can put all your annotation files into **`crowd_raw_batches`**. Accordingly, you should check the `organized_crowd_ans.csv`, `merged_annotation.csv`, and `merged_annotation.json`.

- Organize raw annotation files
    * run `python organize_raw_ans_anno.py`
    * output: `organized_crowd_ans.csv`, `organized_expert_ans.csv`

- Merge the second stage annotation with the first stage annotation
    * run `python merge_annotation.py`
    * output: `merged_annotation.csv`

- Transform CSV to JSON
    * run `python csv2json.py`
    * output: `merged_annotation.json`

