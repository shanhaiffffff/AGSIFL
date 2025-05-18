import os
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import *
import torch
from typing import Tuple

def get_ray_from_depth(d, V=11, dv=10, a0=None, F_W=3 / 8):

    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)

    return rays

def localize4(
        desdf: torch.tensor,
        semantic_desdf: torch.tensor,
        rays: torch.tensor,
        semantic_rays: torch.tensor,
        orn_slice=36,
        return_np=True,
        lambd=40,
        semantic_lambd=40,
        alpha=0.8
) -> Tuple[torch.tensor]:

    # Flip the rays, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    semantic_rays = torch.flip(semantic_rays, [0])

    O = desdf.shape[2]
    V = rays.shape[0]

    # Expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))
    semantic_rays = semantic_rays.reshape((1, 1, -1))

    # Circular pad the desdf and semantic_desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")
    pad_semantic_desdf = F.pad(semantic_desdf, [pad_front, pad_back], mode="circular")

    # Distance-based probability
    dist_prob_vol = torch.stack(
        [
            -torch.norm(pad_desdf[:, :, i: i + V] - rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H, W, O)
    dist_prob_vol = torch.exp(dist_prob_vol / lambd)  # Normalize to probability

    # Semantic-based probability
    semantic_prob_vol = torch.stack(
        [
            -torch.norm(pad_semantic_desdf[:, :, i: i + V] - semantic_rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H, W, O)
    semantic_prob_vol = torch.exp(semantic_prob_vol / semantic_lambd)  # Normalize to probability
    print("desdf shape:", desdf.shape)
    print("semantic_desdf shape:", semantic_desdf.shape)

    # Combine probabilities with dynamic weights
    prob_vol = alpha * dist_prob_vol + (1 - alpha) * semantic_prob_vol

    # Maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # Get the prediction
    pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
    orn = orientations[pred_y, pred_x]
    # From orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))

    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )

def get_ray_from_semantic(semantic_img, V=11, dv=10):

    H, W = semantic_img.shape
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    # Calculate the column indices for each ray
    w = np.tan(angles) * W * (3 / 8) + (W - 1) / 2  # desired width, left to right
    w = np.clip(w, 0, W - 1).astype(int)  # Clip to valid column indices

    # Extract semantic rays
    semantic_rays = np.zeros(V, dtype=int)
    for i, col in enumerate(w):
        if np.any(semantic_img[:, col] == 255):  # Check if any pixel in the column is a door
            semantic_rays[i] = 1  # Door
        else:
            semantic_rays[i] = 0  # Others

    return semantic_rays

def transit(
    prob_vol,
    transition,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):

    H, W, O = list(prob_vol.shape)
    # construction O filters
    filters_trans, filter_rot = get_filters(
        transition,
        O,
        sig_o=sig_o,
        sig_x=sig_x,
        sig_y=sig_y,
        tsize=tsize,
        rsize=rsize,
        resolution=resolution,
    )  # (O, 5, 5), (5,)

    # set grouped 2d convolution, O as channels
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)

    # convolve with the translational filters
    # NOTE: make sure the filter is convolved correctly need to flip
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)

    # convolve with rotational filters
    # reshape as batch
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))  # (HxW, 1, O)
    prob_vol = F.pad(
        prob_vol, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    )
    prob_vol = F.conv1d(
        prob_vol, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    )  # TODO (HxW, 1, O)

    # reshape
    prob_vol = prob_vol.reshape([H, W, O])  # (H, W, O)
    # normalize
    prob_vol = prob_vol / prob_vol.sum()

    return prob_vol


def get_filters(
    transition,
    O=36,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):

    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # calculate center of the gaussian for 36 orientations
    # center for orientation stays the same
    center_o = transition[-1]
    # center_x and center_y depends on the orientation, in total O different, rotate
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O
        * 2
        * torch.pi
    )  # (O,)
    c_th = torch.cos(orns).reshape((O, 1, 1))  # (O, 1, 1)
    s_th = torch.sin(orns).reshape((O, 1, 1))  # (O, 1, 1)
    center_x = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1)
    center_y = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)

    # add uncertainty
    filters_trans = torch.exp(
        -((grid_x - center_x) ** 2) / (sig_x**2) - (grid_y - center_y) ** 2 / (sig_y**2)
    )  # (O, 5, 5)
    # normalize
    filters_trans = filters_trans / filters_trans.sum(-1).sum(-1).reshape((O, 1, 1))

    # rotation filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)

    return filters_trans, filter_rot
