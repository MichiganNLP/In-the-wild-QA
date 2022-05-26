import pandas as pd
import numpy as np

def separate_links_to_channels(links):
    '''
    separate links by channels
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
    df = pd.read_csv("Videos-dropbox-shared-link-Agriculture.csv")
    links = df["link"]
    np.random.seed(1)
    link_groups = separate_links_to_channels(links) # separate links into groups (channels)
    
    # Randomly select videos from the video pool to construct multiples of 5
    all_links = [link for group in link_groups for link in group]
    print(f"there are {len(all_links)} origin videos")
    all_links = all_links + list(np.random.choice(all_links,(5-len(all_links)%5)%5,replace=False))
    
    # randomly select 5 videos for each hit
    np.random.shuffle(all_links)
    input_df = pd.DataFrame(np.array_split(all_links,len(all_links)/5),
                        columns=["video1","video2","video3","video4","video5"])
    input_df.to_csv("whole_Agriculture_input.csv",index=False)
