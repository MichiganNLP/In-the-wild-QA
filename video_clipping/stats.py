import os
import re
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


args_dict = dict(
    select_path='selected_clips',
    ov_path="../youtube-crawler/Videos/",
    output_info_path="stats",
    domain='NaturalDisasters',
)

DOMAINS = ["Agriculture", "Geography", "HumanSurvival", "Military", "NaturalDisasters"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def dm_info(args, domain, plot_all=False):
    chs = os.listdir(f'{args.select_path}/{domain}')

    tt_sel_num = 0

    all_missing_ovs = []

    overtime_vids = []
    undertime_vids = []

    durations = []

    for ch in chs:
        vids = os.listdir(os.path.join(args.select_path, domain, ch))

        # calculate the number of original videos
        ov_chs = os.listdir(f'{args.ov_path}/{domain}/{ch}')
        original_vids = [vid.split('.mp4')[0] for vid in ov_chs]

        exist_sel_clips = os.listdir(f'{args.select_path}/{domain}/{ch}/')
        exist_sel_ov = [cl.split('-clip')[0] if '-clip' in cl else cl.split('-manual')[0] for cl in exist_sel_clips]

        tt_sel_num += len(exist_sel_ov)

        missing_ovs = [vid for vid in original_vids if vid not in exist_sel_ov]

        all_missing_ovs.extend(missing_ovs)

        for vid in exist_sel_clips:
            videopath = f'{args.select_path}/{domain}/{ch}/{vid}'
            clip = VideoFileClip(videopath)

            durations.append(clip.duration)

            conversion = datetime.timedelta(seconds=clip.duration)
            converted_time = str(conversion)
            print(f'{vid}: {converted_time}')
            if clip.duration > 240:
                overtime_vids.append(f'{vid}: {converted_time}')
            if clip.duration < 30:
                undertime_vids.append(f'{vid}: {converted_time}')
    
    if plot_all:
        return durations

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

    plt_duration_hist(domain, durations)
   


def plt_duration_hist(domain, durations, plot_all=False):
    # draw figures of duration distribution for video clips

    fig, ax = plt.subplots()
    if isinstance(durations, list):
        # plt.hist(durations, color=color, ec='black', label=" ".join(re.findall("[A-Z][a-z]*", domain)))
        plt.hist(durations, color=COLORS[DOMAINS.index(domain)], ec='black')
    else:
        assert isinstance(durations, dict)
        # plot all domain distributions
        all_ts = [ts for _, ts in durations.items()]
        plt.hist(all_ts, bins=np.arange(1, 300, 30), alpha=1, color=COLORS, ec='None', stacked=True)
        plt.legend({dm: color for dm, color in zip(DOMAINS, COLORS)})
    fig.canvas.draw()   # make label available
    labels = [item.get_text() for item in ax.get_xticklabels()]

    for idx, label in enumerate(labels):
        labels[idx] = str(round(int(re.sub(u"\u2212", "-", label))/60, 1))
    ax.set_xticklabels(labels)
    # plt.ylabel('Number of clips')
    # plt.xlabel('Durations (min)')
    if not plot_all:
        plt.savefig(f'{args.output_info_path}/{domain}-distribution.pdf')
    else:
        plt.savefig(f'{args.output_info_path}/All-distribution.pdf')


if __name__ == "__main__":
    args = argparse.Namespace(**args_dict)

    if args.domain == "All":
        data = dict()
        durations = list()
        for dm in DOMAINS:
            print(f"For {dm}:\n")
            data[dm] = dm_info(args, dm, plot_all=True)
            durations.extend(data[dm])
            print(f"Existing clips number: {len(data[dm])}\n")
        print(f"Max duration: {round(max(durations)/60, 1)} min = {max(durations)} s,\
        min duration: {round(min(durations)/60, 1)} min = {min(durations)} s, average duration: \
            {round(sum(durations)/(60 * len(durations)), 1)} min = {sum(durations)/len(durations)} s")
        plt_duration_hist(None, data, plot_all=True)
        
    else:
        dm_info(args, args.domain)