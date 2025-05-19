"""
Utils and example of how to generate semantic_desdf for localization
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_semantic_mask(semantic_map_path):

    with tqdm(total=6, desc="Processing progress") as pbar:

        semantic_map = cv2.imread(semantic_map_path)
        pbar.update(1)

        semantic_map_rgb = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2RGB)
        pbar.update(1)


        semantic_mask = np.zeros(semantic_map.shape[:2], dtype=np.uint8)
        pbar.update(1)


        wall_lower = np.array([0, 0, 0])
        wall_upper = np.array([50, 50, 50])
        wall_mask = cv2.inRange(semantic_map_rgb, wall_lower, wall_upper)
        semantic_mask[wall_mask > 0] = 2
        pbar.update(1)


        door_lower1 = np.array([150, 0, 0])
        door_upper1 = np.array([255, 50, 50])
        door_lower2 = np.array([0, 0, 150])
        door_upper2 = np.array([50, 50, 255])
        door_mask1 = cv2.inRange(semantic_map_rgb, door_lower1, door_upper1)
        door_mask2 = cv2.inRange(semantic_map, door_lower2, door_upper2)
        semantic_mask[(door_mask1 + door_mask2) > 0] = 1
        pbar.update(1)


        wall_pixels = np.sum(semantic_mask == 2)
        door_pixels = np.sum(semantic_mask == 1)
        pbar.set_postfix({"Wall pixels": wall_pixels, "Gate Pixel": door_pixels})
        pbar.update(1)

    return semantic_mask

def ray_cast_semantic(occ, semantic_mask, pos, ang, dist_max=500):

    h, w = occ.shape

    occ = np.where((semantic_mask == 1) | (semantic_mask == 2), 255, 0).astype(np.uint8)
    current_pos = pos.copy()
    c = np.cos(ang)
    s = np.sin(ang)

    for step in range(int(dist_max)):
        current_pos[0] += s
        current_pos[1] += c


        if (current_pos[0] < 0 or current_pos[0] >= h or
                current_pos[1] < 0 or current_pos[1] >= w):
            return 0


        if occ[int(current_pos[0]), int(current_pos[1])] > 0:

            return 1 if semantic_mask[int(current_pos[0]), int(current_pos[1])] == 1 else 0

    return 0


def raycast_semantic_desdf(occ, semantic_mask, orn_slice=36, max_dist=10,
                           original_resolution=0.01, resolution=0.1):

    print("\n[3/4] Generating semantic DESDF...")
    ratio = resolution / original_resolution
    H_out = int(occ.shape[0] // ratio)
    W_out = int(occ.shape[1] // ratio)

    semantic_desdf = np.zeros((H_out, W_out, orn_slice), dtype=int)


    for o in tqdm(range(orn_slice), desc="Direction processing", position=0):
        theta = o / orn_slice * np.pi * 2


        for row_out in tqdm(range(H_out), desc="line processing", position=1, leave=False):
            for col_out in range(W_out):
                row_in = row_out * ratio
                col_in = col_out * ratio
                semantic_desdf[row_out, col_out, o] = ray_cast_semantic(
                    occ, semantic_mask,
                    pos=np.array([row_in, col_in]),
                    ang=theta,
                    dist_max=max_dist / original_resolution
                )

    return semantic_desdf


if __name__ == "__main__":
    print("=== Start semantic DESDF generation ===")


    semantic_map_path = os.path.join(
        "/home/zlab/pengshun/AGSIFL-main/data/Gibson Floorplan Localization Dataset/gibson_f/Woonsocket","semantic_map.png")

    semantic_mask = extract_semantic_mask(semantic_map_path)
    occ = cv2.imread(semantic_map_path)[:, :, 0]


    print("\n[2/4] Trimming map boundaries...")
    with tqdm(total=4, desc="Boundary calculation") as pbar:
        l = np.min(np.where(occ == 0)[1]) // 10 * 10
        pbar.update(1)
        r = (np.max(np.where(occ == 0)[1]) // 10 + 1) * 10
        pbar.update(1)
        t = np.min(np.where(occ == 0)[0]) // 10 * 10
        pbar.update(1)
        b = (np.max(np.where(occ == 0)[0]) // 10 + 1) * 10
        pbar.update(1)

    occ = occ[t:b, l:r]
    semantic_mask = semantic_mask[t:b, l:r]
    print(f"Size after cutting: {occ.shape}")


    semantic_desdf = raycast_semantic_desdf(
        occ, semantic_mask, orn_slice=36, max_dist=20,
        original_resolution=0.01, resolution=0.1
    )

    print("\n[4/4] Saving results...")
    desdf = {
        "l": l,
        "t": t,
        "comment": "desdf coordinate to map coordinate: x_map = x_desdf*10 + l, y_map = y_desdf*10 + t",
        "desdf": semantic_desdf
    }

    save_path = "/home/zlab/pengshun/AGSIFL-main/data/Gibson Floorplan Localization Dataset/semantic_desdf/Woonsocket/semantic_desdf.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, desdf)
