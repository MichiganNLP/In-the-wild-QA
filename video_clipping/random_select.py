#!/usr/bin/env python
import argparse
import os
import random

args_dict = {
    "seed": 42,
    "clip_path": "auto-clips",
    "domain": "NaturalDisasters",
    "output_dir": "selected_clips",
}


def main() -> None:
    args = argparse.Namespace(**args_dict)

    random.seed(args.seed)

    chs = os.listdir(os.path.join(args.clip_path, args.domain))

    for ch in chs:
        videos = os.listdir(os.path.join(args.clip_path, args.domain, ch))
        # calculate the number of original videos
        original_videos = {vid.split("-clip")[0] for vid in videos}

        # checking whether we have clips for all the original videos
        # original_videos = list(original_videos)
        # original_videos.sort()
        # print("\n".join(original_videos))
        if not os.path.exists(f"{args.output_dir}/{args.domain}/{ch}/"):
            os.makedirs(f"{args.output_dir}/{args.domain}/{ch}/")

        for ov in original_videos:
            exist_sel_clips = os.listdir(f"{args.output_dir}/{args.domain}/{ch}/")
            exist_sel_ov = [cl.split("-clip")[0] for cl in exist_sel_clips]

            # for each iteration, only select those clips from videos that have not been processed
            # in the output directory
            if ov not in exist_sel_ov:
                clips = [vid for vid in videos if vid.split("-clip")[0] == ov]

                selected_clip = random.choice(clips)

                print(f"For {ov}, there are {len(clips)} clips in total, select {selected_clip}")

                # move the selected videos to path selected clips
                os.rename(os.path.join(args.clip_path, args.domain, ch, selected_clip),
                          os.path.join(args.output_dir, args.domain, ch, selected_clip))


if __name__ == "__main__":
    main()
