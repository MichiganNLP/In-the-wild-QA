import argparse
import json
import os

from pytube import YouTube


def download(url, idx, channel_name, output_path):

    print(f"Downloading {idx} video...")
    try:
        video = YouTube(url)
    except:
        print("Connection Error")

    video.streams
    video.streams.filter(file_extension = "mp4")
    try: 
        # downloading the video 
        video.streams.get_by_itag(18).download(output_path=f'{output_path}/{channel_name}', filename=f'{channel_name}_{idx}.mp4')
    except: 
        print("Downloading Error!") 


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_to_description_links", help="description links, which contains the selected video information", required=True)
    argparser.add_argument("--channel_name", help="channel name")
    argparser.add_argument("--output_path", help="path where the video is put at")
    args = argparser.parse_args()


    with open(args.path_to_description_links, 'r') as f:
        infos = f.readlines()
    infos = [json.loads(info) for info in infos]

    if not os.path.exists(f'{args.output_path}/{args.channel_name}'):
        os.makedirs(f'{args.output_path}/{args.channel_name}')

    for idx, info in enumerate(infos):
        url = info["url"]	
        try:
            download(url, idx, args.channel_name, args.output_path)
        except Exception:
            print(f"An HTTP error occurred")
    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('Finish downloading')


	