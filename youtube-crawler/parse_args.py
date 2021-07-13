import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Common arguments among models
    parser.add_argument('--youtube_url',
        help='link for the channel to crawl, one each time',
        required=True)

    args = parser.parse_args()
    return args
