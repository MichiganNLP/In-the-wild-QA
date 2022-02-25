#!/usr/bin/env python
import os
from collections import defaultdict

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm


def str2t(time: str) -> int:
    return sum(x * int(t) for x, t in zip([60, 1], time.split(":")))


def main() -> None:
    data = defaultdict(list)
    dm_name = None

    with open("manual_clip.txt") as file:
        for d in file:
            tokens = d.split()
            if len(tokens) == 1:
                dm_name = tokens[0]
            elif "(whole video)" in d or not tokens:
                # already manually added
                continue
            else:
                v_name, start, _, end = tokens
                data[dm_name].append([v_name, str2t(start), str2t(end)])

    for dm, items in tqdm(data.items()):
        for item in tqdm(items):
            v_name, start, end = item
            ch, _ = v_name.split("_")
            source = os.path.join("../youtube-crawler/Videos", dm, ch, v_name)
            target = os.path.join("selected_clips", dm, ch, f"{v_name}-manual.mp4")
            ffmpeg_extract_subclip(source, start, end, targetname=target)


if __name__ == "__main__":
    main()
