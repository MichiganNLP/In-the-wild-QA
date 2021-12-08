import json
import os
from typing import Callable, Dict

from overrides import overrides
import PIL.Image
import torch
import torch.utils.data


class WildQaFrameDataset(torch.utils.data.Dataset):
    """Dataset of LifeQA video frames."""

    def __init__(self, transform: Callable = None, videos_data_path: str = 'video_features/frames',
                 check_missing_videos: bool = True) -> None:
        self.transform = transform

        domains = os.listdir(videos_data_path)

        self.video_ids = []

        for domain in domains:
            channels = os.listdir(os.path.join(videos_data_path, domain))
            for channel in channels:
                vids = os.listdir(os.path.join(videos_data_path, domain, channel))
                for vid in vids:
                    self.video_ids.append([domain, channel, vid])

        self.frame_count_by_video_id = {video_id: len(os.listdir(self._video_folder_path(domain, channel, video_id)))
                                        for domain, channel, video_id in self.video_ids}

    @staticmethod
    def _video_folder_path(domain: str, channel: str, video_id: str) -> str:
        return f'video_features/frames/{domain}/{channel}/{video_id}'

    @staticmethod
    def features_file_path(model_name: str, layer_name: str) -> str:
        return f'video_features/features/WildQA_{model_name.upper()}_{layer_name}.hdf5'

    @overrides
    def __getitem__(self, index) -> Dict[str, object]:
        domain, channel, video_id = self.video_ids[index]

        frames = None
        frame_count = self.frame_count_by_video_id[video_id]

        # We're supposing an entire video fits in memory.
        video_folder_path = self._video_folder_path(domain, channel, video_id)
        for i, frame_file_name in enumerate(os.listdir(video_folder_path)):
            frame = PIL.Image.open(os.path.join(video_folder_path, frame_file_name))
            assert frame.mode == 'RGB'

            if self.transform:
                frame = self.transform(frame)

            if frames is None:
                frames = torch.empty((frame_count, *frame.size()))
            frames[i] = frame

        return {
            'id': video_id,
            'frames': frames,
            'frame_count': frame_count
        }

    # @overrides
    def __len__(self) -> int:
        return len(self.video_ids)