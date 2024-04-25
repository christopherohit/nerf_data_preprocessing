# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import glob
import torchvision
from pathlib import Path

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda" if torch.cuda.is_available() else "cpu"
)


def select_significant_keypoints(tracked_keypoints, threshold=0.5):
    """
    Select keypoints with significant flow changes using Laplacian filtering.

    Args:
    tracked_keypoints (torch.Tensor): Tensor of tracked keypoints of shape (num_frames, num_keypoints, 2).
    threshold (float): Threshold value for selecting significant flow changes.

    Returns:
    torch.Tensor: Indices of keypoints with significant flow changes.
    """
    # Convert tracked keypoints to PyTorch tensor
    tracked_keypoints_tensor = torch.tensor(tracked_keypoints)

    # Compute displacement vectors between consecutive frames
    displacement_vectors = tracked_keypoints_tensor[1:] - tracked_keypoints_tensor[:-1]

    # Apply Laplacian filter to displacement vectors
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float
    ).view(1, 1, 3, 3)
    laplacian_filtered_vectors = F.conv2d(
        displacement_vectors.permute(2, 0, 1).unsqueeze(1),
        laplacian_kernel,
        padding=1,
    )

    # Compute magnitude of Laplacian-filtered vectors
    laplacian_magnitudes = torch.norm(laplacian_filtered_vectors, dim=0)

    # Threshold Laplacian magnitudes to select keypoints with significant flow changes
    significant_flow_mask = laplacian_magnitudes > threshold

    result = significant_flow_mask[0, 0]
    for i in range(significant_flow_mask.shape[1]):
        result = torch.logical_or(result, significant_flow_mask[0, i])

    # significant_change_mask = (
    #     torch.norm(tracked_keypoints_tensor[-1] - tracked_keypoints_tensor[0], dim=-1)
    # ) > 14

    # result = torch.logical_and(result, ~significant_change_mask)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        default="./assets/apple.mp4",
        help="path to a imgs",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument("--mask-path", type=str, default=None)

    args = parser.parse_args()

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    segm_mask = (
        np.array(Image.open(os.path.join(args.mask_path))).sum(-1).astype(np.float32)
    )
    segm_mask = (torch.from_numpy(segm_mask)[None, None] > 0).to(torch.float32)

    # print(f"{segm_mask.shape=}")
    # assert False

    window_frames = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = torch.tensor(
            np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
        ).float()[None]  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            segm_mask=segm_mask,
        )

    # Iterating over video frames, processing one window at a time:
    img_dir: str = args.img_path
    assert os.path.exists(img_dir)

    img_paths = list(
        sorted(glob.glob(f"{img_dir}/*.jpg"), key=lambda x: int(Path(x).stem))
    )
    is_first_step = True
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        frame = torchvision.io.read_image(path)
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )

    print("         [INFO] Tracks are computed")

    # save a video with predicted tracks
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE)[None]

    vis = Visualizer(
        save_dir="./saved_videos",
        pad_value=120,
        linewidth=3,
        fps=25,
        mode="optical_flow",
    )
    # vis.visualize(
    #     video,
    #     pred_tracks,
    #     pred_visibility,
    #     query_frame=args.grid_query_frame,
    #     filename="initial",
    # )

    laplacian_filtered_mask = select_significant_keypoints(
        pred_tracks[0].cpu(), threshold=5
    )

    visibility_mask = pred_visibility.squeeze().sum(0) > (len(window_frames) * 3 // 4)

    laplacian_filtered_mask = torch.logical_and(
        laplacian_filtered_mask, visibility_mask.cpu()
    )

    print("         [INFO] Laplacian filter applied.")

    pred_tracks = pred_tracks[:, :, laplacian_filtered_mask]
    pred_visibility = pred_visibility[:, :, laplacian_filtered_mask]

    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=args.grid_query_frame,
        filename="filtered",
    )

    torch.save(pred_tracks, os.path.join(os.path.dirname(img_dir), "keypoints.pt"))
    print("         [INFO] Done!")
