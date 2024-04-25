import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from util import forward_transform


# Placeholder function for initialization stage
def initialize_keypoints(
    keypoints: torch.Tensor,
    euler_angle: torch.Tensor,
    trans: torch.Tensor,
    focal_length: torch.Tensor,
    cxy: torch.Tensor,
):
    num_keypoints = keypoints.shape[1]
    euler_angle = euler_angle.cuda()
    trans = trans.cuda()
    focal_length = focal_length.cuda()
    spatial_coords = torch.randn(
        keypoints.shape[0], num_keypoints, 3, requires_grad=True, device="cuda"
    )

    optimizer = torch.optim.Adam([spatial_coords], lr=0.005)

    losses = []
    for i in tqdm(range(20000)):
        proj_spatial_coords = forward_transform(
            spatial_coords, euler_angle, trans, focal_length, cxy
        )
        print("proj_spatial_coords[:, :, :2]: ", proj_spatial_coords[:, :, :2].shape, keypoints.shape)
        loss_init = F.mse_loss(proj_spatial_coords[:, :, :2], keypoints)
        losses.append(loss_init)

        optimizer.zero_grad()
        loss_init.backward()
        optimizer.step()

        if i % 100 == 0:
            print(sum(losses) / len(losses))
            losses = []

    return spatial_coords, euler_angle, focal_length, trans


# Placeholder function for comprehensive optimization stage
def optimize_keypoints_and_pose(
    keypoints: torch.Tensor,
    id_para: torch.Tensor,
    exp_para: torch.Tensor,
    euler_angle: torch.Tensor,
    trans: torch.Tensor,
    focal_length: torch.Tensor,
    cxy: torch.Tensor,
    spatial_coords: torch.Tensor,
):
    id_para = id_para.new_tensor(id_para.data, device="cuda", requires_grad=True)
    exp_para = exp_para.new_tensor(exp_para.data, device="cuda", requires_grad=True)
    euler_angle = euler_angle.new_tensor(
        euler_angle.data, device="cuda", requires_grad=True
    )
    trans = trans.new_tensor(trans.data, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([euler_angle, trans, spatial_coords], lr=0.0001)

    losses = []
    for i in tqdm(range(10000)):
        proj_spatial_coords = forward_transform(
            spatial_coords, euler_angle, trans, focal_length, cxy
        )

        loss_sec = F.mse_loss(proj_spatial_coords[:, :, :2], keypoints)
        losses.append(loss_sec)

        optimizer.zero_grad()
        loss_sec.backward()
        optimizer.step()

        if i % 100 == 0:
            print(sum(losses) / len(losses))
            losses = []

    return id_para, exp_para, euler_angle, trans, spatial_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle Adjustment of refined Rotation and Translation parameters."
    )
    parser.add_argument("--keypoints-path", type=str, required=True)
    parser.add_argument("--track-params-path", type=str, required=True)
    return parser.parse_args()


def process(keypoints_path: str, track_params_path: str) -> None:
    keypoints = torch.load(keypoints_path).squeeze()

    w, h = 512, 512
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()

    track_params = torch.load(track_params_path)
    id_para = track_params["id"].cuda()
    exp_para = track_params["exp"].cuda()
    euler_angle = track_params["euler"].cuda()
    trans = track_params["trans"].cuda()
    focal_length = track_params["focal"]

    spatial_coords, euler_angle, focal_length, trans = initialize_keypoints(
        keypoints, euler_angle, trans, focal_length, cxy
    )

    print(euler_angle)

    id_para, exp_para, euler_angle, trans, spatial_coords = optimize_keypoints_and_pose(
        keypoints,
        id_para,
        exp_para,
        euler_angle,
        trans,
        focal_length,
        cxy,
        spatial_coords,
    )

    print(euler_angle)
    
    # new track_params
    track_params = {
        "id": id_para.detach().cpu(),
        "exp": exp_para.detach().cpu(),
        "euler": euler_angle.detach().cpu(),
        "trans": trans.detach().cpu(),
        "focal": focal_length.detach().cpu(),
    }

    torch.save(
        track_params,
        os.path.join(os.path.dirname(track_params_path), "track_params_init.pt"),
    )

    torch.save(
        {
            "id": id_para.detach().cpu(),
            "exp": exp_para.detach().cpu(),
            "euler": euler_angle.detach().cpu(),
            "trans": trans.detach().cpu(),
            "focal": focal_length.detach().cpu(),
        },
        track_params_path,
    )


def main() -> None:
    args = parse_args()

    keypoints_path: str = args.keypoints_path
    track_params_path: str = args.track_params_path

    assert os.path.exists(keypoints_path)
    assert os.path.exists(track_params_path)

    process(keypoints_path, track_params_path)


if __name__ == "__main__":
    main()
