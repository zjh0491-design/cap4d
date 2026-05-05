#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from copy import deepcopy
import random
import json
from typing import Union, List
import numpy as np
import torch
from gaussianavatars.scene.dataset_readers import loadCAP4DDataset
from gaussianavatars.scene.cameras import Camera
from gaussianavatars.scene.cap4d_gaussian_model import CAP4DGaussianModel
# from arguments import ModelParams
from gaussianavatars.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from gaussianavatars.utils.general_utils import PILtoTorch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])

            if camera.image_name is not None:
                if camera.image is None:
                    image = Image.open(camera.image_path)
                else:
                    image = camera.image

                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
                arr = arr.squeeze()

                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)

                if arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))

                image = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")

                # ---- from loadCam() and Camera.__init__() ----
                resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

                image = resized_image_rgb[:3, ...]

                if resized_image_rgb.shape[1] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    image *= gt_alpha_mask
                
                camera.original_image = image.clamp(0.0, 1.0)
            else:
                ... # Skip loading image if there is no image path

            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")


class Scene:

    gaussians: CAP4DGaussianModel

    def __init__(
        self, 
        model_path,
        gaussians : CAP4DGaussianModel, 
        source_paths=None,
        target_paths=None,
        shuffle=True, 
        resolution_scales=[1.0],
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = model_path
        self.gaussians = gaussians

        # load dataset
        scene_info = loadCAP4DDataset(
            source_paths=source_paths, 
            target_paths=target_paths,
        )

        # process cameras
        self.train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}
        self.tgt_cameras = {}
        
        if gaussians.binding == None:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        if scene_info.val_cameras:
            camlist.extend(scene_info.val_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.val_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if len(scene_info.train_cameras) > 0:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale)
                print("Loading Validation Cameras")
                self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale)

            if len(scene_info.tgt_cameras) > 0:
                print("Loading Tgt Cameras")
                self.tgt_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.tgt_cameras, resolution_scale)
        
        # process meshes
        if gaussians.binding != None:
            self.gaussians.load_meshes(scene_info.train_meshes, scene_info.test_meshes, scene_info.tgt_meshes)
        
        self.gaussians.create_from_pcd(None, self.cameras_extent)

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale])
    
    def getValCameras(self, scale=1.0):
        return CameraDataset(self.val_cameras[scale])

    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale])

    def getTgtCameras(self, scale=1.0):
        return CameraDataset(self.tgt_cameras[scale])