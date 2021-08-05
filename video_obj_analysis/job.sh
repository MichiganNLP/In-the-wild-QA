CUDA_ID=2

CUDA_VISIBLE_DEVICES=${CUDA_ID} python object_tracker.py \
    --video_path AiirSource-Military_0.mp4 \
    --out_vid_dir detection_info \
    --output_dir processed_videos