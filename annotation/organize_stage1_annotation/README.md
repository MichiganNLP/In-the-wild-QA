# Usage

You can complete this annotation in several times, so some functions are divided into "the first time" version and the "After the first time" version. Choose one accordingly.

- Generate raw results sheet
    * put **all** Amazon Mechanical Turk annotation files into the folder [`raw_batches`](raw_batches/)
    * run `python process_raw_anns.py`

    > **keep the output file in this folder**: `raw_result.csv`

- Generate file for manual review
    * the first time
    ```bash
    csvsql --query \
    "alter table raw_result add column modified_question; \
    alter table raw_result add column modified_answer; \
    select domain, assignment_id,video_link,question,modified_question, \
    correct_answer,modified_answer,evidences_in_min \
    from raw_result" raw_result.csv > to_review.csv
    ```

    * After the first time
    ```bash
    csvsql --query \
    "alter table raw_result add column modified_question; \
    alter table raw_result add column modified_answer; \
    select domain, assignment_id,video_link,question,modified_question, \
    correct_answer,modified_answer,evidences_in_min \
    from raw_result as a \
    where a.assignment_id not in \
    (select assignment_id from processed_reviewed)" \
    raw_result.csv processed_reviewed.csv > to_review.csv
	```
    * for both "the first time" and "After the first time", **after review**, name the reviewed file as `to_review.csv` and put it into this folder

- Organize reviewed "to_review" file
    * the first time:
        * run `./organize_reviewed_first_time.sh`
    * after the first time
        * run `./organize_reviewed_after_first_time.sh`
        * you can delete (not necessary) `processed_reviewed_old.csv` and `processed_reviewed_rm_del_old.csv`

	> **keep the output files in this folder**: `processed_reviewed.csv`, `processed_reviewed_rm_del.csv`

- Transform CSV to JSON file
    * run `python csv2json.py`
    * output: `processed_reviewed_rm_del.json`
