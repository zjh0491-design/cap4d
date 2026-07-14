from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch


def load_motion_features(path: Path) -> torch.Tensor:
    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
    except Exception:
        data = np.load(path, allow_pickle=True)

    if isinstance(data, np.lib.npyio.NpzFile):
        if len(data.files) != 1:
            raise ValueError(f"Cannot infer feature array from npz keys: {data.files}")
        data = data[data.files[0]]
    if isinstance(data, dict):
        for key in ("motion_features", "features", "x"):
            if key in data:
                data = data[key]
                break

    features = torch.as_tensor(data).float()
    if features.ndim == 4:
        if features.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(features.shape)}")
        features = features[0]
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)
    elif features.ndim == 1:
        features = features[None]
    elif features.ndim != 2:
        raise ValueError(f"Unsupported motion feature shape: {tuple(features.shape)}")
    if features.shape[-1] != 512:
        raise ValueError(f"Expected [frames, 512], got {tuple(features.shape)}")
    return features.contiguous()


def load_frame_index(path: Path):
    candidates = [Path(str(path) + ".frames.json"), path.with_suffix(".frames.json")]
    index_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if index_path is None:
        return None, None
    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    image_paths = data if isinstance(data, list) else data.get("image_paths")
    if not isinstance(image_paths, list):
        raise ValueError(f"Unsupported frame index format: {index_path}")
    return data, image_paths


def write_control(path: Path, features: torch.Tensor, frame_index):
    torch.save(features, path)
    if frame_index is not None:
        index_path = Path(str(path) + ".frames.json")
        index_path.write_text(json.dumps(frame_index, indent=2) + "\n", encoding="utf-8")


def main(args):
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_motion_features(input_path)
    frame_index, image_paths = load_frame_index(input_path)
    if frame_index is None and not args.allow_missing_index:
        raise FileNotFoundError(
            f"No .frames.json found next to {input_path}; refusing to build unindexed controls."
        )
    if image_paths is not None and len(image_paths) != features.shape[0]:
        raise ValueError(
            f"Frame index length {len(image_paths)} does not match features {features.shape[0]}."
        )

    prefix = args.prefix or input_path.stem
    zero_path = output_dir / f"{prefix}_zero512.pt"
    shuffle_path = output_dir / f"{prefix}_fixed_shuffle_seed{args.seed}.pt"

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    if features.shape[0] > 1 and torch.equal(permutation, torch.arange(features.shape[0])):
        permutation = permutation.roll(1)

    write_control(zero_path, torch.zeros_like(features), frame_index)
    write_control(shuffle_path, features[permutation], frame_index)
    permutation_path = output_dir / f"{prefix}_fixed_shuffle_seed{args.seed}.permutation.json"
    permutation_path.write_text(
        json.dumps(permutation.tolist(), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"aligned={input_path} shape={tuple(features.shape)}")
    print(f"zero={zero_path}")
    print(f"fixed_shuffle={shuffle_path}")
    print(f"permutation={permutation_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Build indexed zero512 and fixed-shuffle Xnemo controls.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_missing_index", action="store_true", default=False)
    main(parser.parse_args())
