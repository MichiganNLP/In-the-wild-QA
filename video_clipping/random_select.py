import argparse
import os
import random

args_dict = dict(
    seed=42,
    clip_path='auto-clips',
    domain='NaturalDisasters',
    output_dir='selected_clips',
)

args = argparse.Namespace(**args_dict)

random.seed(args.seed)

chs = os.listdir(f'{args.clip_path}/{args.domain}')

for ch in chs:
    vids = os.listdir(os.path.join(args.clip_path, args.domain, ch))
    # calculate the number of original videos
    original_vids = {vid.split('-clip')[0] for vid in vids}

    # checking whether we have clips for all the original videos
    # original_vids = list(original_vids)
    # original_vids.sort()
    # print('\n'.join(original_vids))
    if not os.path.exists(f'{args.output_dir}/{args.domain}/{ch}/'):
        os.makedirs(f'{args.output_dir}/{args.domain}/{ch}/')

    for ov in original_vids:

        exist_sel_clips = os.listdir(f'{args.output_dir}/{args.domain}/{ch}/')
        exist_sel_ov = [cl.split('-clip')[0] for cl in exist_sel_clips]

        # for each iteration, only select those clips from videos that have not been processed 
        # in the output directory
        if ov not in exist_sel_ov:
            clips = [vid for vid in vids if vid.split('-clip')[0] == ov]

            selected_clip = random.choice(clips)

            print(f"For {ov}, there are {len(clips)} clips in total, select {selected_clip}")

            # move the selected vids to path selected clips
            os.rename(f"{args.clip_path}/{args.domain}/{ch}/{selected_clip}",
                      f'{args.output_dir}/{args.domain}/{ch}/{selected_clip}')
