# corresponding channel names
declare -a channel_names=(
    "Disaster-Compilations"
)

Domain=NaturalDisasters

path_to_output=Videos/${Domain}
mkdir -p ${path_to_output}

# get length of an array
arraylength=${#channel_names[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    echo "Downloading videos for channel: ${channel_names[$i]}"
    path_to_description_links=Description/${Domain}/${channel_names[$i]}.jsonl

    python video_downloader.py \
        --path_to_description_links ${path_to_description_links} \
        --channel_name ${channel_names[$i]} \
        --output_path ${path_to_output}
done
