# Youtube video information crawler
The code is adapted from https://github.com/qiisziilbash/YouTube-Channel-Video-Crawler .

## Instruction to use
In order to make selenium work with Firefox
> $ wget https://github.com/mozilla/geckodriver/releases/download/v0.25.0/geckodriver-v0.25.0-linux64.tar.gz -O /tmp/geckodriver.tar.gz \
$ tar -C /opt -xzf /tmp/geckodriver.tar.gz \
$ chmod 755 /opt/geckodriver \
$ ln -fs /opt/geckodriver /usr/bin/geckodriver \
$ ln -fs /opt/geckodriver /usr/local/bin/geckodriver

# Steps to use
* First provide channel links and their names in `crawl_links.sh`, it will crawl the video links and place them under `All-links`
* Run `random_selector.py` to select videos from those channels randomly
* Run `crawl_info.sh` to crawl the description info of those selected videos
* Examine the description and videos manually to filter out unwanted videos, including:
  * videos of person taking interview
  * description of another language (Korean in some cases)
  * Irrelavant videos (Some ads)
* Repeat the steps of running `random_selector.py` and `crawl_info.sh` and changing interation number in the files. Stop until we have expected number of video files for each channel. 

# Result
* links are placed under `All-links` folder.
* Intermediate selected links during the iteration are placed under `Selected-links` folder
* The final list of selected video and their description is placed under `Description`
