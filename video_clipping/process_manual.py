from collections import defaultdict

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

with open(f'manual_clip.txt', 'r') as f:
    raw_data = f.readlines()

def str2t(time):
    return sum(x * int(t) for x, t in zip([60, 1], time.split(":"))) 

data = defaultdict(list)
dm_name = None
for d in raw_data:
    toks = d.split()
    if len(toks) == 1:
        dm_name = toks[0]
    elif "(whole video)" in d or not toks:
        # already manually added
        continue
    else:
        v_name, start, _, end = toks
        data[dm_name].append([v_name, str2t(start), str2t(end)])

for dm, itms in tqdm(iter(data.items())):
    for itm in tqdm(iter(itms)):
        v_name, start, end = itm
        ch, _ = v_name.split("_")
        source = f"../youtube-crawler/Videos/{dm}/{ch}/{v_name}.mp4"
        target = f"selected_clips/{dm}/{ch}/{v_name}-manual.mp4"
        ffmpeg_extract_subclip(source, start, end, targetname=target)
