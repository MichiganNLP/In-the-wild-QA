# generate raw results sheet
`python process_raw_anns.py`

> keep file: `raw_result.csv` 

# generate to review sheet
* the first time
```bash
csvsql --query \
"alter table raw_result add column modified_question; \
alter table raw_result add column modified_answer; \
alter table raw_result add column modified_evidence; \
select domain, assignment_id,video_link,question,modified_question, \
correct_answer,modified_answer,evidences_in_min,modified_evidence \
from raw_result" raw_result.csv > to_review.csv
```

* After the first time
```bash
csvsql --query \
"alter table raw_result add column modified_question; \
alter table raw_result add column modified_answer; \
alter table raw_result add column modified_evidence; \
select domain, assignment_id,video_link,question,modified_question, \
correct_answer,modified_answer,evidences_in_min,modified_evidence \
from raw_result as a \
where a.assignment_id not in \
(select assignment_id from processed_reviewed)" \
raw_result.csv processed_reviewed.csv > to_review.csv
```

# organize reviewed to_review file
> keep file: `processed_reviewed.csv`, `processed_reviewed_rm_del.csv`
* the first time:
`./organize_reviewed_first_time.sh`
* after the first time
`./organize_reviewed_after_first_time.sh`

# trans back to json file
`python csv2json.py`
