
## File usages

* `drive_videos/`
    * `find_url_for_drive.py`: find google drive video ids to share
    * `process.py`: process video links to get the general information of each video clips


- pages
	in annotation_page

- first annotation stage input data
        * `python generate_phase1_input.py -f $your_video_links_csv_file`
        * `$your_video_links_csv_file` should include a column named as 'link'
        * output: first_stage_annotation.csv
	
- question organize -> review
    in organize_stage1_annotation
    
- answer input
- answer organize
human agreement
data analysis
