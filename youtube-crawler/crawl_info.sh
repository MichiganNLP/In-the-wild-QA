#!/bin/bash
## declare an array variable
declare -a urls=(    
    "https://www.youtube.com/c/GungHoVids"
    "https://www.youtube.com/c/AiirSource"
    "https://www.youtube.com/channel/UCG_iY6TpIw4M798sszuG25Q"
    "https://www.youtube.com/c/WarLeaker"
    "https://www.youtube.com/channel/UCT4layPPCzgR_99g5VYYvmQ"
    "https://www.youtube.com/user/UsaMilitaryChannel"
    "https://www.youtube.com/user/MilitaryNotes"
    "https://www.youtube.com/c/SandboxxUs"
)

# corresponding channel names
declare -a channel_names=(
    "Gung-Ho-Vids"
    "AiirSource-Military"
    "Army-military-2018"
    "WarLeaks-Military-Blog"
    "Military-Archive"
    "USA-Military-Channel"
    "MilitaryNotes"
    "Sandboxx"
)

Domain=Military
Iteration=18

mkdir -p Description/${Domain}

# get length of an array
arraylength=${#urls[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    echo "Processing: ${channel_names[$i]}, link: ${urls[$i]}"
    python crawler.py \
        description \
        --links_path Selected-links/${Domain}/${channel_names[$i]}-${Iteration}-selected.txt \
        --youtube_url ${urls[$i]} \
        --domain ${Domain} \
        --channel_name ${channel_names[$i]}
done