
## Usage

* `drive_videos/`
    * `find_url_for_drive.py`: find google drive video ids to share
    * `process.py`: process video links to get the general information of each video clips

- annotation web pages
    * in  `annotation_page`

- generage first stage annotation Amazon Mechanical Turk input file
        * `python generate_stage1_input.py -f $video_links_csv_file`
        * `$video_links_csv_file` should include a column named as 'link'
        * output: `first_stage_annotation.csv`
	
- organize the first stage annotation & manual review
    * see `organize_stage1_annotation`
    
- generage second annotation stage Amazon Mechanical Turk input file
    * run `python generate_stage2_input.py -bs $batch_size`
    * the output can be one or more csv files, back them up into `previous_stage2_inputs`
    * output: `second_stage_annotation_input_{$i}.csv`, each `i` for one Amazon Mechanical Turk "HIT"(batch)
    
- organize the second stage annotation
    * see `organize_stage2_annotation`
    * `organize_stage2_annotation/merged_annotation.csv` and `organize_stage2_annotation/merged_annotation.json` are the final annotation files

- Human agreement calculation & data analysis
    * in `data analysis`
