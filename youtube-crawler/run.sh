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

mkdir -p All-links/${Domain}

# get length of an array
arraylength=${#urls[@]}

# use for loop to read all values and indexes
for (( i=0; i<${arraylength}; i++ ));
do
    echo "Processing: ${channel_names[$i]}, link: ${urls[$i]}"
    python crawler.py \
        --youtube_url ${urls[$i]} > All-links/${Domain}/${channel_names[$i]}.txt
done