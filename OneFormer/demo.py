# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import time
import cv2
import numpy as np
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "OneFormer Demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--input",
        help="Path to an input image or a directory containing images",
    )
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def process_image(demo, img_path, args, logger):
    img = read_image(img_path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img, args.task, img_path)
    logger.info(
        "{}: {} in {:.2f}s".format(
            img_path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    if args.output:
        for k in visualized_output.keys():
            opath = os.path.join(args.output, k)
            os.makedirs(opath, exist_ok=True)
            out_filename = os.path.join(opath, os.path.basename(img_path))
            visualized_output[k].save(out_filename)
    else:
        raise ValueError("Please specify an output path!")

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        input_path = args.input
        if os.path.isfile(input_path):
            # Single image processing
            process_image(demo, input_path, args, logger)
        elif os.path.isdir(input_path):
            # Batch processing for a directory of images
            image_extensions = ['.png', '.jpg', '.jpeg']
            image_files = []
            for root, _, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))

            for img_path in tqdm.tqdm(image_files, disable=not args.output):
                process_image(demo, img_path, args, logger)
        else:
            raise ValueError("Invalid input path! Please provide a valid image file or directory.")
    else:
        raise ValueError("No Input Given")