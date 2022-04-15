#!/usr/bin/env python
"""Script to extract I3D features from video frames."""
import math
import os

import h5py
import torch
import torch.nn
import torch.utils.data
import torchvision
from cached_path import cached_path
from tqdm.auto import tqdm

from src.utils.feature_extraction.i3d import I3D
from src.utils.feature_extraction.wildqa_dataset import WildQaFrameDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1)


def main() -> None:
    dataset = WildQaFrameDataset(transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]))
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=NUM_WORKERS, pin_memory=True,
                                              persistent_workers=NUM_WORKERS > 0)

    model = I3D()

    checkpoint = torch.load(cached_path("https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"))
    model.load_state_dict(checkpoint)

    model.eval()

    model.to(DEVICE)

    stride = 16
    padding = (0, 0, 0, 0, math.ceil((stride - 1) / 2), (stride - 1) // 2)

    with torch.inference_mode():
        with h5py.File(WildQaFrameDataset.features_file_path("i3d", "avg_pool"), "w") as features_file:
            for _, _, video_id in dataset.video_ids:
                video_frame_count = dataset.frame_count_by_video_id[video_id]
                features_file.create_dataset(video_id, shape=(video_frame_count, 1024))
            # tqdm reports an inaccurate ETA if we update it frame by frame.
            for batch in tqdm(data_loader, desc="Extracting I3D features"):
                frames = batch["frames"].to(DEVICE).transpose(1, 2)  # I3D and ReplicationPad3d expect a CTHW order.
                frames = torch.nn.ReplicationPad3d(padding)(frames)

                for i_video, (video_id, video_frame_count) in enumerate(zip(batch["id"], batch["frame_count"])):
                    for i_frame in range(video_frame_count.item()):
                        output = model(frames[[i_video], :, i_frame:i_frame + stride, ...], extract_features=True)
                        features_file[video_id][i_frame] = output[i_video].squeeze().cpu()


if __name__ == "__main__":
    main()
