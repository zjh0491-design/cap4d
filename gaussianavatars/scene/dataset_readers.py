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

from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
import json

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from flowface.flame.utils import batch_rodrigues, OPENCV2PYTORCH3D

from cap4d.datasets.utils import (
    adjust_intrinsics_crop,
    get_crop_mask,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_meshes: dict = {}
    tgt_cameras: list = []


class CVCameraInfo(NamedTuple):
    uid: int
    rt: np.array
    intrinsics: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bg: np.array = np.array([1, 1, 1])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None
    is_pseudo: bool = False
    is_back_view: bool = False
    pseudo_weight: float = 1.0
    source_id: Optional[int] = None
    source_path: Optional[str] = None


def reverse_transform(extr, rot, tra):
    """
    Adjust extrinsics and head rotation to fix head at origin.
    This means that the camera rotates around head instead of head rotating in world coords.
    We need this to get head rotation dependent lighting.
    This is a hack though, technically view and head pose changes will lead to the same lighting effects
    - it looks cool though :)
    """
    T_head = torch.eye(4)[None]
    T_head[:, :3, :3] = batch_rodrigues(torch.tensor(rot)[None])
    T_head[:, :3, 3] = torch.tensor(tra)
    new_extr = torch.tensor(extr).float() @ OPENCV2PYTORCH3D @ T_head[0] @ OPENCV2PYTORCH3D.inverse()
    # Since we rotate camera around head we need to set rot and tra to zero
    new_rot = rot * 0.
    new_tra = tra * 0.

    return new_extr, new_rot, new_tra


def loadCAP4DItem(idx, flame_path, image_path, source_id=None, source_path=None):
    flame_item = dict(np.load(flame_path))

    # we are loading cropped images
    with Image.open(image_path) as img_file:
        image = img_file.copy()

    bg = np.array([1, 1, 1])

    orig_resolution = flame_item["resolutions"][0]
    crop_width, crop_height = image.size
    crop_box = flame_item["crop_box"]

    # adjust intrinsics according to crop box
    fx, fy, cx, cy = [flame_item[key][0, 0] for key in ["fx", "fy", "cx", "cy"]]
    fx, fy, cx, cy = adjust_intrinsics_crop(fx, fy, cx, cy, crop_box, crop_width)

    # if the image is cropped, get outcropping mask
    crop_mask = get_crop_mask(orig_resolution, crop_width, crop_box)

    extr, rot, tra = reverse_transform(
        flame_item["extr"][0],
        flame_item["rot"][0],
        flame_item["tra"][0],
    )

    intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
    )

    flame_out = {
        "shape": flame_item["shape"],
        "expr": flame_item["expr"][0],
        "eye_rot": flame_item["eye_rot"][0],
        "rot": rot,
        "tra": tra,
    }

    cam_info = CVCameraInfo(
        rt=extr,
        intrinsics=intrinsics,
        uid=idx, 
        bg=bg, 
        image=image, 
        image_path=image_path, 
        image_name=image_path.stem, 
        width=crop_width, 
        height=crop_height, 
        timestep=idx, 
        camera_id=idx,
        mask=crop_mask,
        source_id=source_id,
        source_path=str(source_path) if source_path is not None else None,
    )

    return cam_info, flame_out


def _as_array(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype)


def _first_vec3(value, key):
    arr = _as_array(value).reshape(-1, 3)
    if arr.shape[0] == 0:
        raise ValueError(f"Pseudo FLAME field {key} is empty")
    return arr[0]


def _first_time_value(value, key):
    arr = _as_array(value)
    if arr.ndim == 0:
        raise ValueError(f"Pseudo FLAME field {key} is empty")
    if arr.ndim == 1:
        return arr
    if arr.shape[0] == 0:
        raise ValueError(f"Pseudo FLAME field {key} is empty")
    return arr[0]


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _resolve_pseudo_path(path, subject_dir, pseudo_back_dir=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if subject_dir is not None:
        candidate = Path(subject_dir) / path
        if candidate.exists():
            return candidate
    if pseudo_back_dir is not None:
        candidate = Path(pseudo_back_dir) / path
        if candidate.exists():
            return candidate
    return path


def _load_pseudo_mask(mask_path, image_size):
    with Image.open(mask_path) as mask_file:
        mask = np.array(mask_file.convert("L"))
    if mask.shape[:2] != (image_size[1], image_size[0]):
        raise ValueError(
            f"Pseudo mask size mismatch for {mask_path}: "
            f"mask={(mask.shape[1], mask.shape[0])}, image={image_size}"
        )
    return mask > 127


def _collect_images_by_stem(image_dir: Path):
    image_paths = sorted(
        [
            path for path in image_dir.glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
    )
    images_by_stem = {}
    duplicates = []
    for path in image_paths:
        if path.stem in images_by_stem:
            duplicates.append(path.stem)
        images_by_stem[path.stem] = path
    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:20])
        raise ValueError(f"Duplicate image stems in {image_dir}: {duplicate_preview}")
    return images_by_stem


def _paired_cap4d_paths(path: Path):
    flame_paths = sorted(list((path / "flame").glob("*.npz")))
    images_by_stem = _collect_images_by_stem(path / "images")

    if len(flame_paths) == 0:
        raise FileNotFoundError(f"No FLAME files found in {path / 'flame'}")
    if len(images_by_stem) == 0:
        raise FileNotFoundError(f"No images found in {path / 'images'}")

    flame_stems = {flame_path.stem for flame_path in flame_paths}
    image_stems = set(images_by_stem.keys())
    missing_images = sorted(flame_stems - image_stems)
    missing_flames = sorted(image_stems - flame_stems)
    if missing_images or missing_flames:
        details = [
            f"Image/FLAME stem mismatch in {path}: "
            f"images={len(images_by_stem)}, flame={len(flame_paths)}"
        ]
        if missing_images:
            details.append("FLAME files without images: " + ", ".join(missing_images[:20]))
        if missing_flames:
            details.append("Images without FLAME files: " + ", ".join(missing_flames[:20]))
        raise ValueError("\n".join(details))

    img_paths = [images_by_stem[flame_path.stem] for flame_path in flame_paths]
    return flame_paths, img_paths


def _find_pseudo_back_json(source_paths):
    candidates = []
    if source_paths is not None:
        for source_path in source_paths:
            source_path = Path(source_path)
            candidates.extend([
                source_path / "pseudo_back_frames.json",
                source_path.parent / "pseudo_back_frames.json",
                source_path.parent.parent / "pseudo_back_frames.json",
            ])
    candidates.extend(Path("extra").glob("*/pseudo_back_frames.json"))

    found = []
    for candidate in candidates:
        if candidate.exists() and candidate not in found:
            found.append(candidate)
    return found


def _normalize_pseudo_back_paths(pseudo_back_paths, source_paths):
    if pseudo_back_paths is None:
        return _find_pseudo_back_json(source_paths)
    if isinstance(pseudo_back_paths, (str, Path)):
        return [Path(pseudo_back_paths)]
    return [Path(path) for path in pseudo_back_paths]


def loadPseudoBackFrames(pseudo_back_path: Path, cam_id_offset=0, timestep_offset=0):
    pseudo_back_path = Path(pseudo_back_path)
    if not pseudo_back_path.exists():
        raise FileNotFoundError(f"Pseudo back metadata does not exist: {pseudo_back_path}")

    print(f"Loading pseudo back frames from {pseudo_back_path}")
    with open(pseudo_back_path, "r", encoding="utf-8") as f:
        pseudo_data = json.load(f)

    frames = pseudo_data.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"{pseudo_back_path} must contain a list field named 'frames'")

    subject_dir = pseudo_data.get("subject_dir")
    pseudo_back_dir = pseudo_data.get("pseudo_back_dir")
    if subject_dir is not None and pseudo_back_dir is not None:
        pseudo_back_dir = Path(subject_dir) / pseudo_back_dir
    cameras = []
    meshes = []
    for local_idx, frame in enumerate(frames):
        if not frame.get("is_pseudo", False):
            continue

        image_path = _resolve_pseudo_path(frame.get("image_path", frame.get("image")), subject_dir, pseudo_back_dir)
        mask_path = _resolve_pseudo_path(frame.get("mask_path", frame.get("mask")), subject_dir, pseudo_back_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Pseudo image does not exist: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Pseudo mask does not exist: {mask_path}")

        with Image.open(image_path) as img_file:
            image_size = img_file.size
        mask = _load_pseudo_mask(mask_path, image_size)

        flame_params = frame["flame_params"]
        camera_params = frame.get("camera", {})
        intrinsics = None
        if "intrinsics" in camera_params:
            intrinsics = _as_array(camera_params["intrinsics"])
            if intrinsics.shape != (3, 3):
                raise ValueError(
                    f"Pseudo frame {frame.get('name', local_idx)} camera.intrinsics must be 3x3, got {intrinsics.shape}"
                )
        if "extrinsics" not in camera_params:
            raise KeyError(f"Pseudo frame {frame.get('name', local_idx)} is missing camera.extrinsics")
        extr = _as_array(camera_params["extrinsics"])
        if extr.shape == (3, 4):
            extr = np.concatenate(
                [extr, np.array([[0., 0., 0., 1.]], dtype=np.float32)],
                axis=0,
            )
        if extr.shape != (4, 4):
            raise ValueError(
                f"Pseudo frame {frame.get('name', local_idx)} camera.extrinsics must be 4x4 or 3x4, got {extr.shape}"
            )

        if intrinsics is None:
            intrinsics = np.array(
                [[_scalar(flame_params["fx"]), 0, _scalar(flame_params["cx"])],
                 [0, _scalar(flame_params["fy"]), _scalar(flame_params["cy"])],
                 [0, 0, 1]],
                dtype=np.float32,
            )

        rot = _first_vec3(flame_params.get("rot", np.zeros((1, 3))), "rot")
        tra = _first_vec3(flame_params.get("tra", np.zeros((1, 3))), "tra")
        extr, rot, tra = reverse_transform(extr, rot, tra)

        global_idx = cam_id_offset + len(cameras)
        timestep = timestep_offset + len(meshes)
        cam_info = CVCameraInfo(
            rt=extr,
            intrinsics=intrinsics,
            uid=global_idx,
            bg=np.array([1, 1, 1]),
            image=None,
            image_path=image_path,
            image_name=frame.get("name", image_path.stem),
            width=image_size[0],
            height=image_size[1],
            timestep=timestep,
            camera_id=global_idx,
            mask=mask,
            is_pseudo=True,
            is_back_view=bool(frame.get("is_back_view", True)),
            pseudo_weight=float(frame.get("pseudo_weight", 1.0)),
        )

        flame_out = {
            "shape": _as_array(flame_params["shape"]),
            "expr": _first_time_value(flame_params.get("expr", np.zeros((1, 65))), "expr"),
            "eye_rot": _first_vec3(flame_params.get("eye_rot", np.zeros((1, 3))), "eye_rot"),
            "rot": rot,
            "tra": tra,
        }

        cameras.append(cam_info)
        meshes.append(flame_out)

    print(f"Loaded pseudo back frames: {len(cameras)}")
    return cameras, meshes


def readCAP4DImageSet(path: Path, cam_id_offset=0, source_id=None):
    flame_paths, img_paths = _paired_cap4d_paths(path)
    
    cameras = []
    meshes = []

    for frame_id in tqdm(range(len(flame_paths))):        
        camera, mesh = loadCAP4DItem(
            frame_id + cam_id_offset, 
            flame_paths[frame_id], 
            img_paths[frame_id], 
            source_id=source_id,
            source_path=path,
        )
        cameras.append(camera)
        meshes.append(mesh)

    return cameras, meshes


def readCAP4DDrivingSequence(paths: Dict[str, Any], cam_id_offset=0):
    fit_path = paths["animation_path"]

    print(f"Loading target sequence from {fit_path}")
    
    fit = dict(np.load(paths["animation_path"]))

    n_frames = fit["expr"].shape[0]

    if "cam_trajectory_path" in paths and paths["cam_trajectory_path"] is not None:
        cam_traj_path = paths["cam_trajectory_path"]
        print(f"Loading camera trajectory from {cam_traj_path}")
        cam_trajectory = dict(np.load(cam_traj_path))

        extr_list = cam_trajectory["extr"]
        fx_list = cam_trajectory["fx"]
        fy_list = cam_trajectory["fy"]
        cx_list = cam_trajectory["cx"]
        cy_list = cam_trajectory["cy"]
        assert extr_list.shape[0] >= n_frames, "number of frames in the"
        " camera trajectory must be greater or equal to the driving sequence"

        resolution = cam_trajectory["resolution"]
    else:
        # select first camera of driving sequence and repeat (static camera)
        extr_list = fit["extr"][[0]].repeat(n_frames, axis=0)  
        fx_list = fit["fx"][[0]].repeat(n_frames, axis=0)
        fy_list = fit["fy"][[0]].repeat(n_frames, axis=0)
        cx_list = fit["cx"][[0]].repeat(n_frames, axis=0)
        cy_list = fit["cy"][[0]].repeat(n_frames, axis=0)

        resolution = fit["resolutions"][0]

    cameras = []
    meshes = []

    for frame_id in tqdm(range(n_frames)):
        extr, rot, tra = reverse_transform(
            extr_list[frame_id],
            fit["rot"][frame_id],
            fit["tra"][frame_id],
        )

        intrinsics = np.array(
            [[fx_list[frame_id, 0], 0, cx_list[frame_id, 0]],
            [0, fy_list[frame_id, 0], cy_list[frame_id, 0]],
            [0, 0, 1]],
        )

        flame_out = {
            "shape": np.zeros(150),  # shape is set to zero since we don't need it anyways!
            "expr": fit["expr"][frame_id], 
            "eye_rot": fit["eye_rot"][frame_id],
            "rot": rot,
            "tra": tra,
        }

        cam_info = CVCameraInfo(
            rt=extr,
            intrinsics=intrinsics,
            uid=cam_id_offset+frame_id, 
            bg=None, 
            image=None, 
            image_path=None, 
            image_name=None, 
            width=resolution[1], 
            height=resolution[0], 
            timestep=cam_id_offset+frame_id, 
            camera_id=cam_id_offset+frame_id,
            mask=None,
        )

        meshes.append(flame_out)
        cameras.append(cam_info)

    return cameras, meshes


def loadCAP4DDataset(
    source_paths, 
    target_paths: Optional[Dict[str, str]] = None, 
    val_ratio=0.1,
    n_max_val_images=10,
    enable_pseudo_back=False,
    pseudo_back_paths=None,
):
    cameras = []
    meshes = []
    if source_paths is not None:
        for source_id, source_path in enumerate(source_paths):
            source_path = Path(source_path)

            assert source_path.exists(), f"Source path does not exist: {source_path}"

            print(f"Loading dataset from {source_path}")
            cameras_, meshes_ = readCAP4DImageSet(
                source_path,
                cam_id_offset=len(cameras),
                source_id=source_id,
            )

            cameras += cameras_
            meshes += meshes_

    n_frames = len(cameras)
    if n_frames <= 1:
        n_val = 0
    else:
        n_val = max(1, min(n_max_val_images, int(n_frames * val_ratio), n_frames - 1))

    train_cameras = cameras[:-n_val] if n_val > 0 else cameras  # select the last n cameras as validation cams
    train_meshes = meshes
    val_cameras = cameras[-n_val:] if n_val > 0 else []

    pseudo_cameras = []
    pseudo_meshes = []
    if enable_pseudo_back:
        pseudo_paths = _normalize_pseudo_back_paths(pseudo_back_paths, source_paths)
        if len(pseudo_paths) == 0:
            print("WARNING: enable_pseudo_back=True but no pseudo_back_frames.json was found")
        for pseudo_path in pseudo_paths:
            cameras_, meshes_ = loadPseudoBackFrames(
                pseudo_path,
                cam_id_offset=len(cameras) + len(pseudo_cameras),
                timestep_offset=len(meshes) + len(pseudo_meshes),
            )
            pseudo_cameras += cameras_
            pseudo_meshes += meshes_

        train_cameras += pseudo_cameras
        train_meshes += pseudo_meshes

    print("Number of real cameras:", n_frames)
    print("Number of pseudo back cameras:", len(pseudo_cameras))
    print("Number of validation cameras:", len(val_cameras))
    print("Number of train cameras:", len(train_cameras))

    test_cameras = val_cameras
    val_cameras = cameras[:n_val]  # These are training cameras
    test_meshes = []

    tgt_meshes = []
    tgt_cameras = []
    if target_paths is not None:
        tgt_cameras, tgt_meshes = readCAP4DDrivingSequence(
            target_paths, 
            cam_id_offset=len(train_meshes)+len(test_meshes)
        )
    
    scene_info = SceneInfo(
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        val_cameras=val_cameras,
        train_meshes=train_meshes,
        test_meshes=test_meshes,
        nerf_normalization={"radius": 1.},
        ply_path=None,
        tgt_meshes=tgt_meshes,
        tgt_cameras=tgt_cameras,
    )

    ## ...
    return scene_info
