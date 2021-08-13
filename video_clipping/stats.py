import os
import argparse
import datetime
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


args_dict = dict(
    select_path='selected_clips',
    ov_path="../youtube-crawler/Videos/",
    output_info_path="selection_info",
    domain='HumanSurvival',
)

args = argparse.Namespace(**args_dict)


chs = os.listdir(f'{args.select_path}/{args.domain}')

tt_sel_num = 0

all_missing_ovs = []

overtime_vids = []
undertime_vids = []

durations = []

for ch in chs:
    vids = os.listdir(os.path.join(args.select_path, args.domain, ch))

    # calculate the number of original videos
    ov_chs = os.listdir(f'{args.ov_path}/{args.domain}/{ch}')
    original_vids = [vid.split('.mp4')[0] for vid in ov_chs]

    exist_sel_clips = os.listdir(f'{args.select_path}/{args.domain}/{ch}/')
    exist_sel_ov = [cl.split('-clip')[0] for cl in exist_sel_clips]

    tt_sel_num += len(exist_sel_ov)

    missing_ovs = [vid for vid in original_vids if vid not in exist_sel_ov]

    all_missing_ovs.extend(missing_ovs)

    for vid in exist_sel_clips:
        videopath = f'{args.select_path}/{args.domain}/{ch}/{vid}'
        clip = VideoFileClip(videopath)

        durations.append(clip.duration)

        conversion = datetime.timedelta(seconds=clip.duration)
        converted_time = str(conversion)
        print(f'{vid}: {converted_time}')
        if clip.duration > 240:
            overtime_vids.append(f'{vid}: {converted_time}')
        if clip.duration < 30:
            undertime_vids.append(f'{vid}: {converted_time}')

print("Missing original videos:\n\n" + "\n".join(all_missing_ovs))
print("\n")
print(f"Video clips that pass 4 mins:\n" + "\n".join(overtime_vids))
print("\n")
print(f"Video clips that under half min:\n" + "\n".join(overtime_vids))
print("\n")
print(f"Existing clips number: {tt_sel_num}\n")
print(f"Max duration: {round(max(durations)/60, 1)} min = {max(durations)} s,\
 min duration: {round(min(durations)/60, 1)} min = {min(durations)} s, average duration: \
    {round(sum(durations)/(60 * len(durations)), 1)} min = {sum(durations)/len(durations)} s")

# draw figures of duration distribution for video clips

fig, ax = plt.subplots()
plt.hist(durations, color='skyblue', ec='black', label="Human Survival")
fig.canvas.draw()   # make label available
labels = [item.get_text() for item in ax.get_xticklabels()]

for idx, label in enumerate(labels):
    labels[idx] = str(round(int(label)/60, 1))
ax.set_xticklabels(labels)
plt.ylabel('Number of clips')
plt.xlabel('Durations (min)')
plt.legend()
plt.savefig(f'{args.output_info_path}/{args.domain}-distribution.png')