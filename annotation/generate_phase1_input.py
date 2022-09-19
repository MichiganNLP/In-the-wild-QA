import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the input file for the first annotation stage.")
    parser.add_argument("-f","--link-file",metavar="VIDEO LINKS CSV FILE", 
        type=str,default="video_links.csv")

    return parser.parse_args()

def separate_links_to_channels(links):
    '''
    In case there are more than one channel, to separate links by channels
    '''
    link_groups = []
    new_group = []
    for link in links:
        if link is not np.nan:
            new_group.append(link)
        else:
            link_groups.append(new_group)
            new_group = []
    link_groups.append(new_group)
    return link_groups

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.link_file)
    links = df["link"]
    np.random.seed(1)
    link_groups = separate_links_to_channels(links)
    
    # Randomly select videos from the video pool to construct multiples of 5
    all_links = [link for group in link_groups for link in group]
    print(f"there are {len(all_links)} origin videos")
    all_links = all_links + list(np.random.choice(all_links,(5-len(all_links)%5)%5,replace=False))
    
    # randomly select 5 videos for each hit
    np.random.shuffle(all_links)
    input_df = pd.DataFrame(np.array_split(all_links,len(all_links)/5),
                        columns=["video1","video2","video3","video4","video5"])
    input_df.to_csv("first_stage_annotation_input.csv",index=False)
