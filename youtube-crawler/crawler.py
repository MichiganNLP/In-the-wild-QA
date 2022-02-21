import json
import os
import time

import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver

from parse_args import parse_args


def get_video_links(channel_url, verbose, sleep_time):
    videos_url = channel_url + "/videos"

    # if you get error look at readme file for instructions
    driver = webdriver.Firefox()
    driver.get(videos_url)

    time.sleep(sleep_time)

    # scroll dow to the button of the page
    if verbose:
        print("Opening the channel in FireFox and scrolling to the bottom of the page ....")
    while True:
        old_height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script("window.scrollTo(0, " + str(old_height) + ");")
        time.sleep(sleep_time)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")

        if new_height == old_height:
            break

    # parse the html and get the links for the videos
    soup = bs(driver.page_source, "html.parser")
    video_tags = soup.findAll("a", attrs={"class": "yt-simple-endpoint style-scope ytd-grid-video-renderer"})

    if verbose:
        print("##################")
        print("Video links:")
    links = []
    for tag in video_tags:
        if "href" in tag.attrs:
            links.append(tag.attrs["href"])
            if verbose:
                print(tag.attrs["href"])

    driver.close()

    return links


def get_video_info(video_url: str):
    """Gets a YouTube video url and returns its info in a JSON format."""
    information = {"url": video_url}
    response = requests.get(video_url)
    soup = bs(response.content, "html.parser")

    def find_descr(scripts):
        for script in scripts:
            if "var ytInitialPlayerResponse" in script.text:
                return script.text.strip("var ytInitialPlayerResponse = ").strip(";")

    descr = json.loads(find_descr(soup.find_all("script")))
    descr_t = descr.get("microformat").get("playerMicroformatRenderer").get("description")

    if descr_t:
        information["description"] = descr_t.get("simpleText", "")
    else:
        information["description"] = ""

    return information


def crawl_youtube_channel(args, channel_url, verbose=False, sleep_time=3, links_path=None, video_id=1):
    """Gets a YouTube channel url and returns a dictionary containing info about the videos."""
    if links_path:
        with open(links_path) as links_file:
            links = [line.strip("\n") for line in links_file]
    else:
        links = get_video_links(channel_url, verbose=verbose, sleep_time=sleep_time)
        if args.choice == "link":
            with open(args.out_path, "w") as file:
                file.write("\n".join(links))
            return

    unknowns = 0
    counter = 1

    # To indicate new iteration of selection
    with open(f"Description/{args.domain}/{args.channel_name}.jsonl", "a+") as file:
        file.write("\n")

    for link in links:
        information = get_video_info("https://www.youtube.com/" + link)
        video_id += 1
        if information["description"]:
            if verbose:
                print(">>> processing video : "" + str(counter) + "" with description")
        else:
            if verbose:
                print(">>> processing video : "" + str(counter) + "" without description")
            unknowns += 1
        counter += 1

        # incrementally writing crawled information to the file
        with open(f"Description/{args.domain}/{args.channel_name}.jsonl", "a+") as file:
            file.write(json.dumps(information) + "\n")

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(str(unknowns) + " without description")
    print(str(len(links) - unknowns) + " out of " + str(len(links)) + " with description")


def main() -> None:
    args = parse_args()
    # provide the YouTube channel url here
    youtube_url = args.youtube_url
    if args.choice == "link":
        crawl_youtube_channel(args, youtube_url, verbose=True)
    if args.choice == "description":
        if not os.path.exists(f"Description/{args.domain}"):
            os.makedirs(f"Description/{args.domain}")
        crawl_youtube_channel(args, youtube_url, verbose=True, links_path=args.links_path, video_id=args.video_ID)


if __name__ == "__main__":
    main()
