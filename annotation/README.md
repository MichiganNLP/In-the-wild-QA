
## Usage

* `drive_videos/`
    * `find_url_for_drive.py`: find google drive video ids to share
    * `process.py`: process video links to get the general information of each video clips

- Annotation web pages
    * checkout the folder  [`annotation_page`](annotation_page/)
- VQA dataset annotation steps: *first stage annotation -> organize first stage annotation results -> second stage annotation -> organize second stage annotation results & merge with the first stage annotation results*
    - Generate first stage annotation Amazon Mechanical Turk input file
        * run `python generate_stage1_input.py -f $video_links_csv_file`
        * `$video_links_csv_file` should include a column named as 'link'
        * output: `first_stage_annotation.csv`
    - Organize the first stage annotation & manual review
        * checkout the folder [`organize_stage1_annotation`](organize_stage1_annotation/)
    - Generate second stage annotation Amazon Mechanical Turk input files
        * run `python generate_stage2_input.py -bs $batch_size`
        * output: can be one or more files (depends on `$batch_size`) named as `second_stage_annotation_input_{$i}.csv`, each `$i` is for one Amazon Mechanical Turk "HIT"(batch). Back them up into folder [`previous_stage2_inputs`](previous_stage2_inputs/)
    - Organize the second stage annotation
        * checkout the folder [`organize_stage2_annotation`](organize_stage2_annotation/)
        * `organize_stage2_annotation/merged_annotation.csv` and `organize_stage2_annotation/merged_annotation.json` are the final annotation files
- Human agreement calculation & data analysis
    * checkout the folder [`data analysis`](data_analysis/)
