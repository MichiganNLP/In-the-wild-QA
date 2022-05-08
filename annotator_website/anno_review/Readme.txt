# generate raw results sheet
python process_raw_anns.py

# the first time to generate the sheet to review
csvsql --query \
"alter table raw_result add column modified_question; \
alter table raw_result add column modified_answer; \
alter table raw_result add column modified_evidence; \
select domain, assignment_id,video_link,question,modified_question, \
correct_answer,modified_answer,evidences_in_min,modified_evidence \
from raw_result" raw_result.csv > to_review.csv

# After the first time, to generate the sheet to review
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

# organize reviewed to_review file
python process_to_review.py

# put rest information back to reviewed file
# After the first time:
mv processed_reviewed.csv processed_reviewed_old.csv
# For both the first and the later times:
csvsql --query \
"select a.modified_question,a.modified_answer,b.* from \
reviewed as a \
left join \
raw_result as b \
on a.assignment_id=b.assignment_id \
and a.video_link=b.video_link 
and a.question=b.question" reviewed.csv raw_result.csv > processed_reviewed.csv


# remove deleted question
# After the first time:
mv processed_reviewed_rm_del.csv processed_reviewed_rm_del_old.csv
# For both the first and the later times:
csvsql --query \
"select * from \
processed_reviewed \
where modified_question is not '[d]'" processed_reviewed.csv > processed_reviewed_rm_del.csv

# combine processed_reviewed/processed_reviewed_rm_del with previous
csvstack processed_reviewed_old.csv processed_reviewed.csv > processed_reviewed_whole.csv
csvstack processed_reviewed_rm_del_old.csv processed_reviewed_rm_del.csv > processed_reviewed_rm_del_whole.csv
rm processed_reviewed_old.csv processed_reviewed.csv processed_reviewed_rm_del_old.csv processed_reviewed_rm_del.csv
mv processed_reviewed_whole.csv processed_reviewed.csv
mv processed_reviewed_rm_del_whole.csv processed_reviewed_rm_del.csv

# trans back to json file
python csv2json.py
