#!/usr/bin/env python3
"""Audit CAP4D source frame order against Xnemo 512 motion features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def normalized_path(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def sha256_bytes(data: bytes, n: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def checksum_array(array: np.ndarray, n: int = 16) -> str:
    arr = np.ascontiguousarray(array)
    header = f"{arr.shape}|{arr.dtype}".encode("utf-8")
    return sha256_bytes(header + b"\0" + arr.tobytes(), n=n)


def collect_images_by_stem(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(image_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in images:
            duplicates.append(path.stem)
        images[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate image stems in {image_dir}: {sorted(set(duplicates))[:20]}")
    return images


def collect_source_pairs(source_paths: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for source_path_raw in source_paths:
        source_path = Path(source_path_raw)
        flame_dir = source_path / "flame"
        image_dir = source_path / "images"
        flame_paths = sorted(flame_dir.glob("*.npz"))
        images_by_stem = collect_images_by_stem(image_dir)
        if not flame_paths:
            raise FileNotFoundError(f"No FLAME files found in {flame_dir}")
        if not images_by_stem:
            raise FileNotFoundError(f"No images found in {image_dir}")

        flame_stems = {path.stem for path in flame_paths}
        image_stems = set(images_by_stem)
        missing_images = sorted(flame_stems - image_stems)
        missing_flames = sorted(image_stems - flame_stems)
        if missing_images or missing_flames:
            raise ValueError(
                f"Image/FLAME mismatch in {source_path}: "
                f"missing_images={missing_images[:20]} missing_flames={missing_flames[:20]}"
            )

        subject_id = source_path.parent.name
        split_name = source_path.name
        for flame_path in flame_paths:
            rows.append(
                {
                    "subject_id": subject_id,
                    "source_split": split_name,
                    "image_path": images_by_stem[flame_path.stem],
                    "flame_path": flame_path,
                }
            )
    return rows


def load_motion_feature_object(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return np.load(path, allow_pickle=True)


def as_motion_feature_matrix(data) -> np.ndarray:
    if isinstance(data, np.lib.npyio.NpzFile):
        if len(data.files) != 1:
            raise ValueError(f"Cannot infer motion feature array from npz keys: {data.files}")
        data = data[data.files[0]]
    if isinstance(data, dict):
        for key in ("motion_features", "features", "x"):
            if key in data:
                data = data[key]
                break
    tensor = torch.as_tensor(data).float()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor.reshape(tensor.shape[0], -1)
    elif tensor.ndim == 2:
        pass
    elif tensor.ndim == 1:
        tensor = tensor[None]
    else:
        raise ValueError(f"Unsupported motion feature shape: {tuple(tensor.shape)}")
    if tensor.shape[-1] != 512:
        raise ValueError(f"Expected per-frame feature dim 512, got {tensor.shape[-1]}")
    return tensor.cpu().numpy().astype(np.float32)


def load_frame_index(motion_feature_path: Path) -> tuple[Path | None, list[str] | None]:
    candidates = [
        Path(str(motion_feature_path) + ".frames.json"),
        motion_feature_path.with_suffix(".frames.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return candidate, data
            if isinstance(data, dict) and isinstance(data.get("image_paths"), list):
                return candidate, data["image_paths"]
            raise ValueError(
                f"Unsupported frame index format in {candidate}: expected a list "
                "or a dict containing image_paths."
            )
    return None, None


def flame_checksums(path: Path) -> dict[str, str | float | int | None]:
    data = np.load(path, allow_pickle=True)
    expr = np.asarray(data["expr"], dtype=np.float32).reshape(-1)
    eye_rot = np.asarray(data["eye_rot"], dtype=np.float32).reshape(-1)
    rot = np.asarray(data["rot"], dtype=np.float32).reshape(-1)
    tra = np.asarray(data["tra"], dtype=np.float32).reshape(-1)
    shape = np.asarray(data["shape"], dtype=np.float32).reshape(-1)
    offset_source = np.concatenate([shape, expr, eye_rot, rot, tra]).astype(np.float32)
    return {
        "flame_expr_checksum": checksum_array(expr),
        "flame_pose_checksum": checksum_array(np.concatenate([eye_rot, rot, tra]).astype(np.float32)),
        "uv_offset_checksum": None,
        "uv_offset_checksum_type": "not_computed_no_cuda_fallback_flame_shape_expr_eye_rot_rot_tra",
        "uv_offset_source_checksum": checksum_array(offset_source),
        "expr_l2": float(np.linalg.norm(expr)),
        "eye_rot_l2": float(np.linalg.norm(eye_rot)),
        "rot_l2": float(np.linalg.norm(rot)),
        "tra_l2": float(np.linalg.norm(tra)),
        "timestep_id": int(np.asarray(data["timestep_id"]).reshape(-1)[0]) if "timestep_id" in data else None,
    }


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "alignment_table.csv"
    json_path = output_dir / "alignment_summary.json"
    if rows:
        fieldnames = list(rows[0].keys())
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_paths",
        nargs="+",
        default=[
            "examples/output/tesla/reference_images",
            "examples/output/tesla/generated_images_paired",
        ],
    )
    parser.add_argument("--motion_feature_path", default="Xnemo/output/tesla_train_motion.npy")
    parser.add_argument("--output_dir", default="reports/xnemo_phase2a")
    parser.add_argument("--preview_rows", type=int, default=5)
    args = parser.parse_args()

    source_rows = collect_source_pairs(args.source_paths)
    feature_path = Path(args.motion_feature_path)
    features = as_motion_feature_matrix(load_motion_feature_object(feature_path))
    index_path, frame_index = load_frame_index(feature_path)

    if len(source_rows) != features.shape[0]:
        raise ValueError(f"Frame count mismatch: source={len(source_rows)} feature={features.shape[0]}")
    if frame_index is not None and len(frame_index) != len(source_rows):
        raise ValueError(f"Frame index length mismatch: index={len(frame_index)} source={len(source_rows)}")

    table: list[dict] = []
    index_mismatches: list[dict] = []
    feature_checksums: list[str] = []
    for idx, row in enumerate(source_rows):
        feat = features[idx]
        feat_checksum = checksum_array(feat)
        feature_checksums.append(feat_checksum)
        image_path = Path(row["image_path"])
        flame_path = Path(row["flame_path"])
        observed_path = frame_index[idx] if frame_index is not None else None
        expected_norm = normalized_path(image_path)
        observed_norm = normalized_path(observed_path) if observed_path is not None else None
        index_matches = observed_norm is None or observed_norm == expected_norm
        if not index_matches:
            index_mismatches.append(
                {
                    "frame_id": idx,
                    "expected": expected_norm,
                    "observed": observed_norm,
                }
            )
        record = {
            "frame_id": idx,
            "subject_id": row["subject_id"],
            "source_split": row["source_split"],
            "image_path": str(image_path.resolve()),
            "flame_path": str(flame_path.resolve()),
            "feature_index": idx,
            "frame_index_path": observed_path,
            "frame_index_matches": index_matches,
            "xnemo512_checksum": feat_checksum,
            "xnemo512_mean": float(feat.mean()),
            "xnemo512_std": float(feat.std()),
            "xnemo512_l2": float(np.linalg.norm(feat)),
            "xnemo512_zero": bool(np.linalg.norm(feat) < 1e-8),
        }
        record.update(flame_checksums(flame_path))
        table.append(record)

    duplicate_features = len(feature_checksums) - len(set(feature_checksums))
    stale_runs = 0
    for prev, curr in zip(feature_checksums, feature_checksums[1:]):
        if prev == curr:
            stale_runs += 1

    summary = {
        "source_paths": [str(Path(path).resolve()) for path in args.source_paths],
        "motion_feature_path": str(feature_path.resolve()),
        "frame_index_file": str(index_path.resolve()) if index_path else None,
        "num_frames": len(table),
        "feature_shape": list(features.shape),
        "frame_index_mismatch_count": len(index_mismatches),
        "frame_index_mismatch_preview": index_mismatches[: args.preview_rows],
        "zero_feature_count": int(sum(row["xnemo512_zero"] for row in table)),
        "duplicate_feature_count": int(duplicate_features),
        "adjacent_duplicate_feature_count": int(stale_runs),
        "unique_feature_checksum_count": int(len(set(feature_checksums))),
        "feature_global_mean": float(features.mean()),
        "feature_global_std": float(features.std()),
        "feature_global_min": float(features.min()),
        "feature_global_max": float(features.max()),
        "uv_offset_checksum_status": "not_computed_no_cuda",
        "note": (
            "uv_offset_checksum is null because this node has no CUDA and the CAP4D model "
            "constructs CUDA tensors during UV rasterization; uv_offset_source_checksum hashes "
            "shape+expr+eye_rot+rot+tra from the aligned FLAME file."
        ),
    }

    write_outputs(table, summary, Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {Path(args.output_dir) / 'alignment_table.csv'}")
    print(f"Wrote {Path(args.output_dir) / 'alignment_summary.json'}")


if __name__ == "__main__":
    main()
