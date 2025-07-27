import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
from attrdict import AttrDict
from torch.utils.data import DataLoader
from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *


def calculate_furniture_ratio(furniture_mask):

    return np.count_nonzero(furniture_mask == 0) / furniture_mask.size


def calculate_depth_rays_difference(rays):

    if isinstance(rays, torch.Tensor):
        rays = rays.cpu().numpy()
    diff = np.abs(rays[:, None] - rays[None, :])
    return np.mean(diff)



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


def evaluate_filtering():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/Gibson Floorplan Localization Dataset",
        help="path of the dataset",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./logs", help="path of the checkpoints"
    )
    parser.add_argument(
        "--evol_path",
        type=str,
        default=None,
        help="path to save the tracking evolution figures",
    )
    parser.add_argument(
        "--traj_len", type=int, default=100, help="length of the trajectory"
    )
    args = parser.parse_args()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    net_type = "comp"

    # paths
    dataset_dir = os.path.join(args.dataset_path, "gibson_t")
    depth_dir = args.dataset_path
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")
    semantic_desdf_path = os.path.join(args.dataset_path, "semantic_desdf")  # Path to semantic_desdf folder
    evol_path = args.evol_path

    # instanciate dataset
    traj_l = args.traj_len
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    test_set = TrajDataset(
        dataset_dir,
        split.test,
        L=traj_l,
        depth_dir=depth_dir,
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=True,
    )

    # logs
    log_error = True
    log_timing = True

    # parameters
    L = 3  # number of the source frames
    D = 128  # number of the depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # camera intrinsic, focal length / image width
    orn_slice = 36  # number of discretized orientations

    mv_net_pl = mv_depth_net_pl(D=D, d_hyp=d_hyp, F_W=F_W)
    mono_net_pl = depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D, F_W=F_W)
    comp_net = comp_d_net_pl.load_from_checkpoint(
        checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
        mv_net=mv_net_pl.net,
        mono_net=mono_net_pl.encoder,
        L=L,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
        D=D,
        F_W=F_W,
        use_pred=True,
    ).to(device)
    comp_net.eval()  # this is needed to disable batchnorm


    input_size = 2
    hidden_size = 16
    output_size = 1

    mlp_checkpoint_path = os.path.join(log_dir, "mlp_checkpoint.pth")
    mlp = MLP(input_size, hidden_size, output_size).to(device)
    mlp.load_state_dict(torch.load(mlp_checkpoint_path))
    mlp.eval()

    # get desdf and semantic_desdf for the scene
    print("load desdf and semantic_desdf ...")
    desdfs = {}
    semantic_desdfs = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 10] = 10  # truncate

        semantic_desdfs[scene] = np.load(
            os.path.join(semantic_desdf_path, scene, "semantic_desdf.npy"),
            allow_pickle=True
        ).item()

    # get the ground truth pose file
    print("load poses and maps ...")
    maps = {}
    gt_poses = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        # load map
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h = occ.shape[0]
        w = occ.shape[1]

        # single trajectory
        poses = np.zeros([0, 3], dtype=np.float32)
        # get poses
        poses_file = os.path.join(dataset_dir, scene, "poses.txt")

        # read poses
        with open(poses_file, "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]

        traj_len = len(poses_txt)
        traj_len -= traj_len % traj_l
        for state_id in range(traj_len):
            # get pose
            pose = poses_txt[state_id].split(" ")
            x = float(pose[0])
            y = float(pose[1])
            th = float(pose[2])
            # from world coordinate to map coordinate
            x = x / 0.01 + w / 2
            y = y / 0.01 + h / 2

            poses = np.concatenate(
                (poses, np.expand_dims(np.array((x, y, th), dtype=np.float32), 0)),
                axis=0,
            )

        gt_poses[scene] = poses

    # record stats
    RMSEs = []
    success_10 = []  # Success @ 1m
    success_5 = []  # Success @ 0.5m
    success_3 = []  # Success @ 0.3m
    success_2 = []  # Success @ 0.2m

    matching_time = 0
    iteration_time = 0
    feature_extraction_time = 0
    n_iter = 0

    # loop the over scenes
    for data_idx in tqdm.tqdm(range(len(test_set))):

        data = test_set[data_idx]
        # get the scene name according to the data_idx
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        # get desdf and semantic_desdf
        desdf = desdfs[scene]
        semantic_desdf = semantic_desdfs[scene]

        # get reference pose in map coordinate and in scene coordinate
        poses_map = gt_poses[scene][
                      idx_within_scene * traj_l: idx_within_scene * traj_l + traj_l, :
                      ]

        # transform to desdf frame
        gt_pose_desdf = poses_map.copy()
        gt_pose_desdf[:, 0] = (gt_pose_desdf[:, 0] - desdf["l"]) / 10
        gt_pose_desdf[:, 1] = (gt_pose_desdf[:, 1] - desdf["t"]) / 10

        imgs = torch.tensor(data["imgs"], device=device).unsqueeze(0)
        poses = torch.tensor(data["poses"], device=device).unsqueeze(0)

        # set prior as uniform distribution
        prior = torch.tensor(
            np.ones_like(desdf["desdf"]) / desdf["desdf"].size, device=imgs.device
        ).to(torch.float32)

        pred_poses_map = []

        for t in range(traj_l - L):
            start_iter = time.time()
            feature_extraction_start = time.time()
            # form input
            input_dict = {
                "ref_img": imgs[:, t + L, :, :, :],
                "src_img": imgs[:, t: t + L, :, :, :],
                "ref_pose": poses[:, t + L, :],
                "src_pose": poses[:, t: t + L, :],
                "ref_mask": None,  # no masks because the dataset has zero roll pitch
                "src_mask": None,  # no masks because the dataset has zero roll pitch
            }

            # inference
            pred_dict = comp_net.comp_d_net(input_dict)
            pred_depths = pred_dict["d_comp"]

            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()

            # get rays from depth
            pred_rays = get_ray_from_depth(pred_depths)
            pred_rays = torch.tensor(pred_rays, device=device)

            # Get door mask for the reference image
            door_mask_dir = os.path.join(args.dataset_path, "gibson_t", scene, "door_mask")
            frame_index = idx_within_scene * traj_l + t + L
            door_mask_path = os.path.join(door_mask_dir, f"{frame_index:05d}.png")
            print("Loading door mask from:", door_mask_path)


            if not os.path.exists(door_mask_path):
                raise FileNotFoundError(f"Door mask file not found: {door_mask_path}")

            door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE)
            if door_mask is None:
                raise ValueError(f"Failed to load door mask from: {door_mask_path}")

            # Get semantic rays
            semantic_rays = get_ray_from_semantic(door_mask)

            # Convert to torch tensors
            semantic_rays = torch.tensor(semantic_rays, dtype=torch.float32).to(device)


            furniture_mask_dir = os.path.join(args.dataset_path, "gibson_t", scene, "furniture_mask")
            furniture_mask_path = os.path.join(furniture_mask_dir, f"{frame_index:05d}.png")
            furniture_mask = cv2.imread(furniture_mask_path, cv2.IMREAD_GRAYSCALE)
            if furniture_mask is None:
                raise ValueError(f"Failed to load furniture mask from: {furniture_mask_path}")


            furniture_ratio = calculate_furniture_ratio(furniture_mask)


            depth_rays_difference = calculate_depth_rays_difference(pred_rays)


            mlp_input = torch.tensor([[furniture_ratio, depth_rays_difference]], dtype=torch.float32).to(device)
            alpha_mlp = mlp(mlp_input).item()

            feature_extraction_end = time.time()

            matching_start = time.time()
            # use the prediction to localize, produce observation likelihood
            likelihood, likelihood_2d, _, likelihood_pred = localize4(
                torch.tensor(desdf["desdf"]).to(prior.device),
                torch.tensor(semantic_desdf["desdf"]).to(prior.device),
                pred_rays.to(prior.device),
                semantic_rays.to(prior.device),
                return_np=False,
                lambd=40,
                semantic_lambd=40,
                alpha=alpha_mlp
            )
            # multiply with the prior
            posterior = prior * likelihood.to(prior.device)

            # reduce the posterior along orientation for 2d visualization
            posterior_2d, orientations = torch.max(posterior, dim=2)

            posterior_2d_np = posterior_2d.detach().cpu().numpy()
            posterior_max = posterior_2d_np.max()

            high_prob_thresh = 0.8 * posterior_max
            high_prob_area = (posterior_2d_np > high_prob_thresh).sum()
            total_area = posterior_2d_np.size
            high_prob_ratio = high_prob_area / total_area

            sorted_probs = np.sort(posterior_2d_np.flatten())[::-1]
            top1_idx = int(0.01 * len(sorted_probs))
            if top1_idx > 0:
                top1_mean = sorted_probs[:top1_idx].mean()
                bottom99_mean = sorted_probs[top1_idx:].mean()
                mean_diff = top1_mean - bottom99_mean

            from scipy.stats import kurtosis
            flatten_posterior = posterior_2d_np.flatten()
            kurt = kurtosis(flatten_posterior)

            if high_prob_ratio < 0.01 and mean_diff > 0.2 and kurt > 10:
                alpha = 1

            else:
                alpha = alpha_mlp

            likelihood, likelihood_2d, _, likelihood_pred = localize4(
                torch.tensor(desdf["desdf"]).to(prior.device),
                torch.tensor(semantic_desdf["desdf"]).to(prior.device),
                pred_rays.to(prior.device),
                semantic_rays.to(prior.device),
                return_np=False,
                lambd=40,
                semantic_lambd=40,
                alpha=alpha
            )
            matching_end = time.time()

            posterior = prior * likelihood.to(prior.device)

            # reduce the posterior along orientation for 2d visualization
            posterior_2d, orientations = torch.max(posterior, dim=2)

            # compute prior_2d for visualization
            prior_2d, _ = torch.max(prior, dim=2)

            # maximum of the posterior as result
            pose_y, pose_x = torch.where(posterior_2d == posterior_2d.max())
            if pose_y.shape[0] > 1:
                pose_y = pose_y[0].unsqueeze(0)
                pose_x = pose_x[0].unsqueeze(0)
            orn = orientations[pose_y, pose_x]

            # from orientation indices to radians
            orn = orn / orn_slice * 2 * torch.pi
            pose = torch.cat((pose_x, pose_y, orn)).detach().cpu().numpy()

            pose_in_map = pose.copy()
            pose_in_map[0] = pose_in_map[0] * 10 + desdf["l"]
            pose_in_map[1] = pose_in_map[1] * 10 + desdf["t"]

            pred_poses_map.append(pose_in_map)

            if evol_path is not None:
                # plot posterior 2d
                fig = plt.figure(0, figsize=(20, 20))
                fig.clf()
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(
                    posterior_2d.detach().cpu().numpy(), origin="lower", cmap="coolwarm"
                )
                ax.quiver(
                    pose[0],
                    pose[1],
                    np.cos(pose[2]),
                    np.sin(pose[2]),
                    color="blue",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.quiver(
                    gt_pose_desdf[t + L, 0],
                    gt_pose_desdf[t + L, 1],
                    np.cos(gt_pose_desdf[t + L, 2]),
                    np.sin(gt_pose_desdf[t + L, 2]),
                    color="green",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.axis("off")
                ax.set_title(str(t) + " posterior")

                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(likelihood_2d, origin="lower", cmap="coolwarm")
                ax.set_title(str(t) + " likelihood")
                ax.axis("off")
                ax.quiver(
                    likelihood_pred[0],
                    likelihood_pred[1],
                    np.cos(likelihood_pred[2]),
                    np.sin(likelihood_pred[2]),
                    color="blue",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.quiver(
                    gt_pose_desdf[t + L, 0],
                    gt_pose_desdf[t + L, 1],
                    np.cos(gt_pose_desdf[t + L, 2]),
                    np.sin(gt_pose_desdf[t + L, 2]),
                    color="green",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )

                if not os.path.exists(
                        os.path.join(evol_path, "pretty_filter", str(data_idx))
                ):
                    os.makedirs(os.path.join(evol_path, "pretty_filter", str(data_idx)))
                fig.savefig(
                    os.path.join(
                        evol_path, "pretty_filter", str(data_idx), str(t) + ".png"
                    )
                )

            # transition
            # use ground truth to compute transitions, use relative poses
            if t + L == traj_l - 1:
                continue
            current_pose = poses[0, t + L, :]
            next_pose = poses[0, t + L + 1, :]

            transition = get_rel_pose(current_pose, next_pose)
            prior = transit(
                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7, rsize=7
            )

            end_iter = time.time()
            matching_time += matching_end - matching_start
            feature_extraction_time += feature_extraction_end - feature_extraction_start
            iteration_time += end_iter - start_iter
            n_iter += 1

        if log_error:
            pred_poses_map = np.stack(pred_poses_map)
            # record success rate, from map to global
            last_errors = (
                    ((pred_poses_map[-5:, :2] - poses_map[-5:, :2]) ** 2).sum(axis=1)
                    ** 0.5
            ) * 0.01
            # compute RMSE
            RMSE = (
                           ((pred_poses_map[-5:, :2] - poses_map[-5:, :2]) ** 2)
                           .sum(axis=1)
                           .mean()
                   ) ** 0.5 * 0.01
            RMSEs.append(RMSE)
            print("last_errors", last_errors)
            if all(last_errors < 1):
                success_10.append(True)
            else:
                success_10.append(False)

            if all(last_errors < 0.5):
                success_5.append(True)
            else:
                success_5.append(False)

            if all(last_errors < 0.3):
                success_3.append(True)
            else:
                success_3.append(False)

            if all(last_errors < 0.2):
                success_2.append(True)
            else:
                success_2.append(False)

    if log_error:
        RMSEs = np.array(RMSEs)
        success_10 = np.array(success_10)
        success_5 = np.array(success_5)
        success_3 = np.array(success_3)
        success_2 = np.array(success_2)

        print("============================================")
        print("1.0 success rate : ", success_10.sum() / len(test_set))
        print("0.5 success rate : ", success_5.sum() / len(test_set))
        print("0.3 success rate : ", success_3.sum() / len(test_set))
        print("0.2 success rate : ", success_2.sum() / len(test_set))
        print("mean RMSE succeeded : ", RMSEs[success_10].mean())
        print("mean RMSE all : ", RMSEs.mean())

    if log_timing:
        feature_extraction_time = feature_extraction_time / n_iter
        matching_time = matching_time / n_iter
        iteration_time = iteration_time / n_iter

        print("============================================")
        print("feature_extraction_time : ", feature_extraction_time)
        print("matching_time : ", matching_time)
        print("iteration_time : ", iteration_time)


if __name__ == "__main__":
    evaluate_filtering()
