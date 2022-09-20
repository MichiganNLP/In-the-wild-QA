#!/bin/bash

# organize reviewed to_review.csv
python process_to_review.py

# put rest information back to reviewed file
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
csvsql --query \
"select * from \
processed_reviewed \
where modified_question is not '[d]'" processed_reviewed.csv > processed_reviewed_rm_del.csv
