#!/usr/bin/env bash
## declare an array variable
declare -a urls=(    
    "https://www.youtube.com/channel/UCMqrWfhtN_CP903slwfmqwQ"
)

# corresponding channel names
declare -a channel_names=(
    "Natural-Disaster"
)

Domain=NaturalDisasters
Iteration=0

mkdir -p Description/${Domain}

# get length of an array
arraylength=${#urls[@]}

# Selected-links/${Domain}/${channel_names[$i]}-${Iteration}-selected.txt \

for (( i=0; i<${arraylength}; i++ ));
do
    echo "Processing: ${channel_names[$i]}, link: ${urls[$i]}"
    python crawler.py \
        description \
        --links_path filtered-links/${Domain}/${channel_names[$i]}.txt \
        --youtube_url ${urls[$i]} \
        --domain ${Domain} \
        --channel_name ${channel_names[$i]}
done