#!/usr/bin/env python
"""Script to extract ResNet features from video frames."""
import argparse
import math

import cv2 as cv
import h5py
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torchvision
from overrides import overrides
from src.utils.feature_extraction.c3d import C3D
from src.utils.feature_extraction.i3d import I3D
from src.utils.feature_extraction.wildqa_dataset import WildQaFrameDataset
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_NORMALIZATION_PARAMS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def pretrained_c3d() -> torch.nn.Module:
    c3d = C3D(pretrained=True)
    c3d.eval()
    for param in c3d.parameters():
        param.requires_grad = False
    return c3d


def pretrained_i3d() -> torch.nn.Module:
    i3d = I3D(pretrained=True)
    i3d.eval()
    for param in i3d.parameters():
        param.requires_grad = False
    return i3d


class Identity(torch.nn.Module):
    @overrides
    def forward(self, input_):
        return input_


def save_resnet_features():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**IMAGENET_NORMALIZATION_PARAMS),
    ])
    dataset = WildQaFrameDataset(transform=transform)

    resnet = pretrained_resnet152().to(DEVICE)
    resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.

    with h5py.File(WildQaFrameDataset.features_file_path('resnet', 'res5c'), 'w') as res5c_features_file, \
            h5py.File(WildQaFrameDataset.features_file_path('resnet', 'pool5'), 'w') as pool5_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
            pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None

        def avg_pool_hook(_module, input_, _output):
            nonlocal res5c_output
            res5c_output = input_[0]

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        # tqdm reports an inaccurate ETA if we update it frame by frame.
        for batch in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting ResNet features"):
            video_ids = batch['id']
            frames = batch['frames'].to(DEVICE)

            for video_id, video_frames in zip(video_ids, frames):
                frame_batch_size = 32
                for start_index in range(0, len(video_frames), frame_batch_size):
                    end_index = min(start_index + frame_batch_size, len(video_frames))
                    frame_ids_range = range(start_index, end_index)
                    frame_batch = video_frames[frame_ids_range]
                    avg_pool_value = resnet(frame_batch)

                    res5c_features_file[video_id][frame_ids_range] = res5c_output.cpu()
                    pool5_features_file[video_id][frame_ids_range] = avg_pool_value.cpu()


def save_resof_features():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        # Same as ResNet's, but without normalizing for ImageNet yet.
    ])
    dataset = WildQaFrameDataset(transform=transform)

    resnet = pretrained_resnet152().to(DEVICE)
    resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.

    filter_size = 16
    padding = (0, 0, 0, 0, math.ceil((filter_size - 1) / 2), (filter_size - 1) // 2)

    imagenet_normalization = torchvision.transforms.Normalize(**IMAGENET_NORMALIZATION_PARAMS)

    with h5py.File(WildQaFrameDataset.features_file_path('resof', 'pool5'), 'w') as features_file:
        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        # tqdm reports an inaccurate ETA if we update it frame by frame.
        for batch in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting ResOF features"):
            video_ids = batch['id']
            video_frame_counts = batch['frame_count']
            frames = batch['frames']

            frames = frames.transpose(1, 2)  # ReplicationPad3d expects (N, C, T, H, W)
            frames = torch.nn.ReplicationPad3d(padding)(frames)
            frames = frames.permute(0, 2, 3, 4, 1)  # OpenCV expects (H, W, C)

            flow_images = torch.empty((frames.shape[0], frames.shape[1] - 1, *frames.shape[2:]))
            hsv = np.empty_like(flow_images.numpy(), dtype=np.uint8)
            hsv[..., 1] = 255

            for i_video, video_frame_count in enumerate(video_frame_counts):
                for i_frame_pair in range(video_frame_count.item() - 1):
                    bw_frame1 = cv.cvtColor(frames[i_video, i_frame_pair].numpy(), cv.COLOR_RGB2GRAY)
                    bw_frame2 = cv.cvtColor(frames[i_video, i_frame_pair + 1].numpy(), cv.COLOR_RGB2GRAY)

                    flow = cv.calcOpticalFlowFarneback(bw_frame1, bw_frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[i_video, i_frame_pair, ..., 0] = ang * 180 / np.pi / 2
                    hsv[i_video, i_frame_pair, ..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

                    flow_image = cv.cvtColor(hsv[i_video, i_frame_pair], cv.COLOR_HSV2RGB) / 255.0
                    flow_images[i_video, i_frame_pair] = torch.from_numpy(flow_image)

            flow_images = flow_images.permute(0, 1, 4, 2, 3)  # ResNet and the transformations expect (N, C, H, W)

            for i_video, (video_id, video_frame_count) in enumerate(zip(video_ids, video_frame_counts)):
                for i_frame_pair in range(video_frame_count.item() - 1):
                    flow_images[i_video, i_frame_pair] = imagenet_normalization(flow_images[i_video, i_frame_pair])

                for i_frame in range(video_frame_count.item()):
                    frame_window = flow_images[i_video, i_frame:i_frame + filter_size - 1, ...]
                    features_file[video_id][i_frame] = resnet(frame_window.to(DEVICE)).max(dim=0)[0].cpu()


def save_c3d_features():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**IMAGENET_NORMALIZATION_PARAMS),
    ])
    dataset = WildQaFrameDataset(transform=transform)

    c3d = pretrained_c3d().to(DEVICE)
    filter_size = 16
    padding = (0, 0, 0, 0, math.ceil((filter_size - 1) / 2), (filter_size - 1) // 2)

    with h5py.File(WildQaFrameDataset.features_file_path('c3d', 'fc6'), 'w') as fc6_features_file, \
            h5py.File(WildQaFrameDataset.features_file_path('c3d', 'conv5b'), 'w') as conv5b_features_file:
        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            fc6_features_file.create_dataset(video_id, shape=(video_frame_count, 4096))
            conv5b_features_file.create_dataset(video_id, shape=(video_frame_count, 1024, 7, 7))

        # tqdm reports an inaccurate ETA if we update it frame by frame.
        for batch in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting C3D features"):
            video_ids = batch['id']
            video_frame_counts = batch['frame_count']
            frames = batch['frames'].to(DEVICE)
            frames = frames.transpose(1, 2)  # C3D and ReplicationPad3d expect (N, C, T, H, W)
            frames = torch.nn.ReplicationPad3d(padding)(frames)

            for i_video, (video_id, video_frame_count) in enumerate(zip(video_ids, video_frame_counts)):
                for i_frame in range(video_frame_count.item()):
                    fc6_output, conv5b_output = \
                        c3d.extract_features(frames[[i_video], :, i_frame:i_frame + filter_size, ...])
                    fc6_features_file[video_id][i_frame] = fc6_output[i_video].cpu()
                    conv5b_features_file[video_id][i_frame] = conv5b_output[i_video].reshape(1024, 7, 7).cpu()


def save_i3d_features():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**IMAGENET_NORMALIZATION_PARAMS),
    ])
    dataset = WildQaFrameDataset(transform=transform)

    i3d = pretrained_i3d().to(DEVICE)
    filter_size = 16
    padding = (0, 0, 0, 0, math.ceil((filter_size - 1) / 2), (filter_size - 1) // 2)

    with h5py.File(WildQaFrameDataset.features_file_path('i3d', 'avg_pool'), 'w') as features_file:
        for _, _, video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            features_file.create_dataset(video_id, shape=(video_frame_count, 1024))
        # tqdm reports an inaccurate ETA if we update it frame by frame.
        for batch in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting I3D features"):
            video_ids = batch['id']
            video_frame_counts = batch['frame_count']
            frames = batch['frames'].to(DEVICE)
            frames = frames.transpose(1, 2)  # I3D and ReplicationPad3d expects (N, C, T, H, W)
            frames = torch.nn.ReplicationPad3d(padding)(frames)

            for i_video, (video_id, video_frame_count) in enumerate(zip(video_ids, video_frame_counts)):
                for i_frame in range(video_frame_count.item()):
                    output = i3d.extract_features(frames[[i_video], :, i_frame:i_frame + filter_size, ...])
                    features_file[video_id][i_frame] = output[i_video].squeeze().cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video features.")
    parser.add_argument('network', choices=['resnet', 'resof', 'c3d', 'i3d'])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.network == 'resnet':
        save_resnet_features()
    elif args.network == 'resof':
        save_resof_features()
    elif args.network == 'c3d':
        save_c3d_features()
    elif args.network == 'i3d':
        save_i3d_features()
    else:
        raise ValueError("Network type not supported.")


if __name__ == '__main__':
    main()
