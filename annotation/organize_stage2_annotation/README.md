## Usage
In the original paper setting, we emploied both 'experts' and 'crowdworkers' in the second stage annotation. Put your Amazon Mechanical Turk annotation files into `expert_raw_batches` and `crowd_raw_batches` accordingly. If your annotation doesn't differentiate between 'experts' and 'crowdworkers', you can put all your annotation files into **`crowd_raw_batches`**. Accordingly, you should check the `organized_crowd_ans.csv`, `merged_annotation.csv`, and `merged_annotation.json`.

- organize raw annotation files
    * `python organize_raw_ans_anno.py`
    * output: `organized_crowd_ans.csv`, `organized_expert_ans.csv`

- merge the second stage annotation with the first stage annotation
    * `python merge_annotation.py`
    * output: `merged_annotation.csv`

- csv to json
    * `python csv2json.py`
    * output: `merged_annotation.json`
