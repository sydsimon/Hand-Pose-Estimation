import copy
import json
import os

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_util import *


class ego4dDataset(Dataset):
    """
    Implementation of Ego-Exo4D dataset for hand pose estimation. 
    Return cropped hand image, ground truth heatmap and valid joint flag as output.
    """
    def __init__(self, cfg, split, transform=None):
        self.split = split
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}. Can only be [train, val, test]"
        self.pixel_std = 200  # Pixel std to define scale factor for image resizing
        self.undist_img_dim = np.array([512, 512])  # Size of undistorted aria image
        self.image_dim = cfg["image_dim"] # Length of input image to the model
        self.heatmap_dim = cfg["heatmap_dim"]
        self.img_dir = os.path.join(cfg["img_dir"], split)
        gt_anno_path = os.path.join(cfg["anno_dir"], f"ego_pose_gt_anno_{self.split}.json")
        self.db = self.load_all_data(gt_anno_path)
        self.transform = transform

    def __getitem__(self, idx):
        """
        Return transformed images, 2d hand kpts, corresponding heatmap, valid joint flag and metadata.
        """
        curr_db = copy.deepcopy(self.db[idx])
        image_size = np.array([self.image_dim, self.image_dim])
        heatmap_size = np.array([self.heatmap_dim, self.heatmap_dim])

        # Define parameters for affine transformation of hand image
        c, s = xyxy2cs(*curr_db["bbox"], self.undist_img_dim, self.pixel_std)
        r = 0
        trans = get_affine_transform(c, s, r, image_size)
        ### 1. Load image
        metadata = curr_db["metadata"]
        img_path = os.path.join(
            self.img_dir,
            f"{metadata['take_name']}",
            f"{metadata['frame_number']:06d}.jpg",
        )
        img = imageio.imread(img_path, pilmode="RGB")
        # Get affine transformed hand image
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.image_dim), int(self.image_dim)),
            flags=cv2.INTER_LINEAR,
        )
        # Apply Pytorch transform if needed
        if self.transform:
            input = self.transform(input)

        ### 2. Generate ground truth 2D kpts heatmap
        kpts_tran = affine_transform(curr_db["joints_2d"], trans)
        kpts_hm, new_valid_flag = generate_target(kpts_tran, 
                                                  curr_db["valid_flag"],
                                                  image_size,
                                                  heatmap_size)
        kpts_hm = torch.from_numpy(kpts_hm)

        ### 3. Generate valid joints flag
        kpts_weight = torch.from_numpy(new_valid_flag)

        # Record meta info for debugging check
        meta = curr_db["metadata"]

        return input, kpts_hm, kpts_weight, meta

    def __len__(self):
        return len(self.db)

    def load_all_data(self, gt_anno_path):
        """
        Store each valid hand's annotation per frame separately as a dictionary 
        with following key:
            - joints_2d: (N,2)
            - valid_flag: (N,)
            - bbox: (4,)
            - metadata
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))

        # Load gt annotation
        all_frame_anno = []
        for _, curr_take_anno in gt_anno.items():
            for _, curr_f_anno in curr_take_anno.items():
                for hand_order in ["right", "left"]:
                    single_hand_anno = {}
                    if len(curr_f_anno[f"{hand_order}_hand"]) != 0:
                        single_hand_anno["joints_2d"] = np.array(
                            curr_f_anno[f"{hand_order}_hand"]
                        )
                        single_hand_anno["valid_flag"] = np.array(
                            curr_f_anno[f"{hand_order}_hand_valid"]
                        )
                        single_hand_anno["bbox"] = np.array(
                            curr_f_anno[f"{hand_order}_hand_bbox"]
                        )
                        single_hand_anno["metadata"] = curr_f_anno["metadata"]
                        all_frame_anno.append(single_hand_anno)
        return all_frame_anno

