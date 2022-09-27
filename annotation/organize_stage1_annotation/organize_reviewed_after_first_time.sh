#!/bin/bash

# organize reviewed to_review.csv
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

rm reviewed.csv

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
rm processed_reviewed.csv processed_reviewed_rm_del.csv
# rm processed_reviewed_old.csv processed_reviewed.csv processed_reviewed_rm_del_old.csv processed_reviewed_rm_del.csv
mv processed_reviewed_whole.csv processed_reviewed.csv
mv processed_reviewed_rm_del_whole.csv processed_reviewed_rm_del.csv

