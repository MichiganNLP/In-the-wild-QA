#!/bin/bash
## declare an array variable
declare -a urls=(    
    "https://www.youtube.com/c/RealAgricultureMedia"
    "https://www.youtube.com/c/MNMillennialFarmer"
    "https://www.youtube.com/c/WelkerFarmsInc"
    "https://www.youtube.com/c/HowFarmsWork"
    "https://www.youtube.com/c/PetersonFarmBros"
    "https://www.youtube.com/user/farmmarketing"
    "https://www.youtube.com/c/Olly%E2%80%99sFarmLtd"
    "https://www.youtube.com/c/HamiltonvilleFarm"
)

# corresponding channel names
declare -a channel_names=(
    "RealAgriculture"
    "Millennial-Farmer"
    "Welker-Farms-Inc"
    "How-Farms-Work"
    "Peterson-Farm-Bros"
    "John-Suscovich"
    "Olly's-Farm"
    "Hamiltonville-Farm"
)

Domain=Agriculture
Iteration=3

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