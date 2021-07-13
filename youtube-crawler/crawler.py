import json
import time

import requests
from selenium import webdriver
from bs4 import BeautifulSoup as bs

from parse_args import parse_args


def get_video_links(channel_url, verbose, sleep_time):
    videos_url = channel_url + '/videos'

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
    video_tags = soup.findAll('a', attrs={'class': 'yt-simple-endpoint style-scope ytd-grid-video-renderer'})

    if verbose:
        print("##################")
        print("Video links:")
    links = []
    for tag in video_tags:
        if 'href' in tag.attrs:
            links.append(tag.attrs['href'])
            if verbose:
                print(tag.attrs['href'])
    
    driver.close()
    
    return links


def get_video_info(video_url):
    """ gets a youtube video url and returns its info in a json format"""
    information = {'url': video_url}
    response = requests.get(video_url)
    soup = bs(response.content, "html.parser")

    description_tag = soup.find('p', id='eow-description')
    if description_tag:
        information['description'] = description_tag.text
    else:
        information['description'] = ''

    return information


def get_article_title(description):
    """ gets a description of a video in 2 min paper channel and returns the title of article"""

    keyword = 'The paper "'
    try:
        if keyword in description:
            description = description[description.index(keyword) + len(keyword):]
            title = description[0:description.index('"')]
        else:
            title = 'unknown'
    except:
        title = 'unknown'
    return title


def crawl_youtube_channel(channel_url, verbose=False, sleep_time=3, links_path=None):
    """ gets a Youtube channel url and returns a dictionary containing info about the videos"""

    if links_path:
        links = []
        links_file = open(links_path, "r")
        lines = links_file.readlines()
        for line in lines:
            links.append(line)
        links_file.close()
    else:
        links = get_video_links(channel_url, verbose=verbose, sleep_time=sleep_time)

    # result = {}
    # video_ID = 1
    # unknowns = 0
    # counter = 1

    # for link in links:
    #     information = get_video_info("https://www.youtube.com/" + link)
    #     information['article'] = get_article_title(information['description'])
    #     del information['description']
    #     if information['article'] != 'unknown':
    #         result[video_ID] = information
    #         video_ID += 1
    #         if verbose:
    #             print('>>> processing video : ' + str(counter) + ' success')
    #     else:
    #         if verbose:
    #             print('>>> processing video : ' + str(counter) + ' Failed')
    #         unknowns += 1
    #     counter += 1
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # print(str(unknowns) + ' failure')
    # print(str(len(links) - unknowns) + ' out of ' + str(len(links)) + ' success')
    # return result


if __name__ == '__main__':
    args = parse_args()
    # provide the youtube channel url here
    youtube_url = args.youtube_url
    crawl_youtube_channel(youtube_url, verbose=True)
    # data = crawl_youtube_channel(youtube_url, verbose=True)
    
    # with open('Data/' + youtube_url.split('/')[-1] + '.txt', 'w') as outfile:
    #     json.dump(data, outfile, indent=4)
