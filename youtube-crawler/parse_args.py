import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog="choice")

    subparsers = parser.add_subparsers(help="sub-command help", dest="choice")

    # Common arguments among models
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("--youtube_url", help="link for the channel to crawl, one each time")

    # crawl video links
    link_parser = subparsers.add_parser("link", parents=[parent_parser])

    link_parser.add_argument("--out_path", help="path for the output file")

    # crawl descriptions
    desc_parser = subparsers.add_parser("description", parents=[parent_parser])
    desc_parser.add_argument("--links_path", help="path for the file containing video links")
    desc_parser.add_argument("--domain", help="domain for the video")
    desc_parser.add_argument("--channel_name", help="channel name for the list")
    desc_parser.add_argument("--video_id", help="video ID to start crawling, start from 1", default=1)

    return parser.parse_args()
