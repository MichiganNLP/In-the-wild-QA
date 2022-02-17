# Standard PySceneDetect imports:
import argparse
import os

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scenedetect import SceneManager, VideoManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector


def find_scenes(video_path, threshold=16.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def decide_split(video_path):
    scenes = find_scenes(video_path)

    tt_p = 0
    start = None
    clips = []
    for idx, (ss, se) in enumerate(scenes):
        if tt_p == 0:
            start = ss
        p = (se - ss).get_seconds()
        tt_p += p
        if tt_p > 60:
            clips.append((start, se))
            tt_p = 0
        elif idx == len(scenes) - 1:
            # last short clips
            clips.append((start, se))

    return clips


def clip_video(video_path, output_dir, output_t_dir):
    clips = decide_split(video_path)
    fn = os.path.basename(video_path).split('.mp4')[0]
    for idx, (ss, se) in enumerate(clips):
        dirname = os.path.dirname(video_path)
        ffmpeg_extract_subclip(video_path, ss.get_seconds(), se.get_seconds(),
                               targetname=f'{output_dir}/{fn}-clip-{idx}.mp4')
    with open(f'{output_t_dir}/{fn}-timecode.txt', 'w') as f:
        for ss, se in clips:
            f.write(f'{ss.get_timecode()}-{se.get_timecode()}\n')
    with open(f'{output_t_dir}/{fn}-seconds.txt', 'w') as f:
        for ss, se in clips:
            f.write(f'{ss.get_seconds()}-{se.get_seconds()}\n')
    with open(f'{output_t_dir}/{fn}-frames.txt', 'w') as f:
        for ss, se in clips:
            f.write(f'{ss}-{se}\n')


args_dict = dict(
    video_path='../youtube-crawler/Videos/',
    output_vid_dir='auto-clips/',
    output_t_dir='auto-clips-info/'
)

args = argparse.Namespace(**args_dict)

rd = args.video_path

domains = [os.path.join(rd, o) for o in os.listdir(rd)
           if os.path.isdir(os.path.join(rd, o))]

for domain in domains:
    chs = [os.path.join(domain, c) for c in os.listdir(domain)]
    for ch in chs:
        vids = [os.path.join(ch, v) for v in os.listdir(ch)]
        domain_name = os.path.basename(domain)
        ch_name = os.path.basename(ch)
        output_vid_dir = os.path.join(args.output_vid_dir, domain_name, ch_name)
        output_t_dir = os.path.join(args.output_t_dir, domain_name, ch_name)

        if not os.path.exists(output_vid_dir):
            os.makedirs(output_vid_dir)

        if not os.path.exists(output_t_dir):
            os.makedirs(output_t_dir)

        for vid in vids:
            clip_video(vid, output_vid_dir, output_t_dir)
