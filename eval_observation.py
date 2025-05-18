import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml
import cv2
import numpy as np
from attrdict import AttrDict
import torch.nn.functional as F
from typing import Tuple
from modules.comp.comp_d_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *
from modules.mv.mv_depth_net_pl import *
from modules.mono.depth_net_pl import *

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        min_val = 0.7
        max_val = 0.9
        out = min_val + out * (max_val - min_val)
        return out

def calculate_furniture_ratio(furniture_mask):

    return np.count_nonzero(furniture_mask == 0) / furniture_mask.size

def calculate_depth_rays_difference(rays):

    if isinstance(rays, torch.Tensor):
        rays = rays.cpu().numpy()
    diff = np.abs(rays[:, None] - rays[None, :])
    return np.mean(diff)


def evaluate_observation(mlp, args):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # Parameters
    L = 3  # Number of the source frames
    D = 128  # Number of depth planes
    d_min = 0.1  # Minimum depth
    d_max = 15.0  # Maximum depth
    d_hyp = -0.2  # Depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # Camera intrinsic, focal length / image width

    # Paths
    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    depth_dir = dataset_dir
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")
    semantic_desdf_path = os.path.join(args.dataset_path, "semantic_desdf")  # Path to semantic_desdf folder

    # Instanciate dataset
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    test_set = GridSeqDataset(
        dataset_dir,
        split.test,
        L=L,
        depth_dir=depth_dir,
        depth_suffix="depth160",  # Only comp network uses depth160
        add_rp=False,
        roll=0,
        pitch=0,
    )

    # Create model (only comp network)
    comp_net = comp_d_net_pl.load_from_checkpoint(
        checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
        mv_net=mv_depth_net_pl(D=D, d_hyp=d_hyp).net,
        mono_net=depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D).encoder,
        L=L,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
        D=D,
        F_W=F_W,
        use_pred=True,
    ).to(device)
    comp_net.eval()  # Disable batchnorm

    # Load desdf and semantic_desdf for the scene
    print("Loading desdf and semantic_desdf ...")
    desdfs = {}
    semantic_desdfs = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        # Load desdf
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 10] = 10  # Truncate

        semantic_desdfs[scene] = np.load(os.path.join(semantic_desdf_path, scene, "semantic_desdf.npy"),
                                         allow_pickle=True
                                         ).item()
    # Load the ground truth poses
    print("Loading poses and maps ...")
    maps = {}
    gt_poses = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        # Load map
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h = occ.shape[0]
        w = occ.shape[1]

        # Get poses
        with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]
            traj_len = len(poses_txt)
            poses = np.zeros([traj_len, 3], dtype=np.float32)
            for state_id in range(traj_len):
                pose = poses_txt[state_id].split(" ")
                # From world coordinate to map coordinate
                x = float(pose[0]) / 0.01 + w / 2
                y = float(pose[1]) / 0.01 + h / 2
                th = float(pose[2])
                poses[state_id, :] = np.array((x, y, th), dtype=np.float32)

            gt_poses[scene] = poses

    # Record the accuracy
    acc_record = []
    acc_orn_record = []
    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        # Get the scene name according to the data_idx
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]

        # Get idx within scene
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        # Get desdf and semantic_desdf
        desdf = desdfs[scene]
        semantic_desdf = semantic_desdfs[scene]

        # Get reference pose in map coordinate and in scene coordinate
        ref_pose_map = gt_poses[scene][idx_within_scene * (L + 1) + L, :]
        src_pose_map = gt_poses[scene][
                       idx_within_scene * (L + 1): idx_within_scene * (L + 1) + L, :
                       ]
        ref_pose = data["ref_pose"]
        src_pose = data["src_pose"]

        data["ref_pose"] = torch.tensor(ref_pose, device=device).unsqueeze(0)
        data["src_pose"] = torch.tensor(src_pose, device=device).unsqueeze(0)

        # Transform to desdf frame
        gt_pose_desdf = ref_pose_map.copy()
        gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) / 10
        gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) / 10

        # Get observation
        ref_img = data["ref_img"]  # (C, H, W)
        src_img = data["src_img"]  # (L, C, H, W)

        # get ground truth roll and pitch
        # do the gravity alignment
        # compute the attention mask
        ref_mask = None  # no masks because the dataset has zero roll pitch

        ref_img_torch = torch.tensor(ref_img, device=device).unsqueeze(0)
        ref_mask_torch = None
        data["ref_img"] = ref_img_torch
        data["ref_mask"] = ref_mask_torch

        src_mask = None  # no masks because the dataset has zero roll pitch

        src_img_torch = torch.tensor(src_img, device=device).unsqueeze(0)
        data["src_img"] = src_img_torch
        src_mask_torch = None
        data["src_mask"] = src_mask_torch

        # Get door mask for the reference image
        door_mask_dir = os.path.join(args.dataset_path, args.dataset, scene, "door_mask")
        scene_index = f"{idx_within_scene:05d}"
        frame_index = L
        door_mask_path = os.path.join(
            door_mask_dir, f"{scene_index}-{frame_index}.png"
        )
        print("Loading door mask from:", door_mask_path)

        if not os.path.exists(door_mask_path):
            raise FileNotFoundError(f"Door mask file not found: {door_mask_path}")

        door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE)
        if door_mask is None:
            raise ValueError(f"Failed to load door mask from: {door_mask_path}")

        # Get semantic rays
        semantic_rays = get_ray_from_semantic(door_mask)

        # Convert to torch tensors
        semantic_rays = torch.tensor(semantic_rays, dtype=torch.float32)

        furniture_mask_dir = os.path.join(args.dataset_path, args.dataset, scene, "furniture_mask")
        furniture_mask_path = os.path.join(furniture_mask_dir, f"{scene_index}-{frame_index}.png")
        furniture_mask = cv2.imread(furniture_mask_path, cv2.IMREAD_GRAYSCALE)
        if furniture_mask is None:
            raise ValueError(f"Failed to load furniture mask from: {furniture_mask_path}")

        furniture_ratio = calculate_furniture_ratio(furniture_mask)

        # Inference (only comp network)
        pred_dict = comp_net.comp_d_net(data)
        pred_depths = pred_dict["d_comp"].squeeze(0).detach().cpu().numpy()

        # Get distance-based rays
        pred_rays = get_ray_from_depth(pred_depths)
        pred_rays = torch.tensor(pred_rays, device="cpu")

        depth_rays_difference = calculate_depth_rays_difference(pred_rays)

        mlp_input = torch.tensor([[furniture_ratio, depth_rays_difference]], dtype=torch.float32).to(device)
        alpha = mlp(mlp_input).item()

        # Localize with the desdf and semantic_desdf using the prediction
        prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize4(
            torch.tensor(desdf["desdf"]),
            torch.tensor(semantic_desdf["desdf"]),
            pred_rays,
            semantic_rays,
            lambd=40,
            semantic_lambd=40,
            alpha=alpha
        )

        # Calculate accuracy
        acc = np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.1
        acc_record.append(acc)
        acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
        acc_orn_record.append(acc_orn)

    acc_record = np.array(acc_record)
    acc_orn_record = np.array(acc_orn_record)
    print("1m recall = ", np.sum(acc_record < 1) / acc_record.shape[0])
    print("0.5m recall = ", np.sum(acc_record < 0.5) / acc_record.shape[0])
    print("0.1m recall = ", np.sum(acc_record < 0.1) / acc_record.shape[0])
    print(
        "1m 30 deg recall = ",
        np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30))
        / acc_record.shape[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--net_type",
        type=str,
        default="comp",
        choices=["comp"],  # Only use comp network
        help="type of the network to evaluate. Only 'comp' is supported.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gibson_f",
        choices=["gibson_f", "gibson_g"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/Gibson Floorplan Localization Dataset",
        help="path of the dataset",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./logs", help="path of the checkpoints"
    )
    args = parser.parse_args()

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # Parameters
    L = 3  # Number of the source frames
    D = 128  # Number of depth planes
    d_min = 0.1  # Minimum depth
    d_max = 15.0  # Maximum depth
    d_hyp = -0.2  # Depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # Camera intrinsic, focal length / image width

    # Paths
    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    depth_dir = dataset_dir
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")
    semantic_desdf_path = os.path.join(args.dataset_path, "semantic_desdf")  # Path to semantic_desdf folder

    # Instanciate dataset
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    test_set = GridSeqDataset(
        dataset_dir,
        split.test,
        L=L,
        depth_dir=depth_dir,
        depth_suffix="depth160",  # Only comp network uses depth160
        add_rp=False,
        roll=0,
        pitch=0,
    )

    # Create model (only comp network)
    comp_net = comp_d_net_pl.load_from_checkpoint(
        checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
        mv_net=mv_depth_net_pl(D=D, d_hyp=d_hyp).net,
        mono_net=depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D).encoder,
        L=L,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
        D=D,
        F_W=F_W,
        use_pred=True,
    ).to(device)
    comp_net.eval()  # Disable batchnorm
    input_size = 2
    hidden_size = 16
    output_size = 1

    mlp_checkpoint_path = os.path.join(log_dir, "mlp_checkpoint.pth")
    mlp = MLP(input_size, hidden_size, output_size).to(device)
    mlp.load_state_dict(torch.load(mlp_checkpoint_path))
    mlp.eval()
    evaluate_observation(mlp, args)