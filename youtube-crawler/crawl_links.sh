#!/usr/bin/env bash
## declare an array variable
declare -a urls=(    
    "https://www.youtube.com/c/AiirSource"
    "https://www.youtube.com/c/WarLeaker"
    "https://www.youtube.com/channel/UCT4layPPCzgR_99g5VYYvmQ"
    "https://www.youtube.com/user/UsaMilitaryChannel"
    "https://www.youtube.com/user/MilitaryNotes"
    "https://www.youtube.com/c/SandboxxUs"
)

# corresponding channel names
declare -a channel_names=(
    "AiirSource-Military"
    "WarLeaks-Military-Blog"
    "Military-Archive"
    "USA-Military-Channel"
    "MilitaryNotes"
    "Sandboxx"
)

Domain=Military

mkdir -p All-links/${Domain}

# get length of an array
arraylength=${#urls[@]}

# Crawl the links for videos of a channel
for (( i=0; i<${arraylength}; i++ ));
do
    echo "Processing: ${channel_names[$i]}, link: ${urls[$i]}"
    python crawler.py \
        link \
        --youtube_url ${urls[$i]} \
        --out_path All-links/${Domain}/${channel_names[$i]}.txt
done
