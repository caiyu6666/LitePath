#!/usr/bin/env python3
"""Reproducible preprocessing and prepared-HDF5 inference benchmark for LitePath.

The ``preprocess`` subcommand measures coordinate extraction and packed-HDF5
materialization per WSI (optionally repeated). The independent ``inference``
subcommand reuses that immutable cache for every method and repeat. Inference computation remains
defined as cached inference wall time minus exposed DataLoader waiting time.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import gc
import hashlib
import io
import importlib.util
import json
import logging
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = SCRIPT_DIR
sys.path.insert(0, str(INFERENCE_ROOT))
os.chdir(INFERENCE_ROOT)

from datasets import PatchDataset  # noqa: E402
from litepath_deploy import ModelDeployment  # noqa: E402
from models import DAttention, get_custom_transformer  # noqa: E402
from preprocessing.create_patches_fp import adjust_size, estimate_best_seg_level  # noqa: E402
from preprocessing.extract_images_and_pack2h5 import read_images  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402


LOGGER = logging.getLogger("litepath_latency")


MODE_METADATA = {
    "aps": {"method": "LitePath-APS", "encoder": "LiteFM", "all_patches": False},
    "uniform": {
        "method": "LitePath-Uniform-2000",
        "encoder": "LiteFM",
        "all_patches": False,
    },
    "litefm-full": {"method": "LiteFM-Full", "encoder": "LiteFM", "all_patches": True},
    "virchow2-full": {
        "method": "Virchow2-Full",
        "encoder": "Virchow2",
        "all_patches": True,
    },
}


def mode_metadata(mode: str) -> Dict[str, Any]:
    return MODE_METADATA[mode]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def run_command(command: Sequence[str], timeout: int = 15) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = completed.stdout.strip()
        return output or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root_candidate = INFERENCE_ROOT / path
    if root_candidate.exists():
        return root_candidate.resolve()
    return (manifest_dir / path).resolve()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    slides = payload["slides"] if isinstance(payload, dict) else payload
    if not isinstance(slides, list) or not slides:
        raise ValueError("Manifest must contain a non-empty 'slides' list")

    resolved: List[Dict[str, Any]] = []
    for entry in slides:
        if "wsi_path" not in entry:
            raise ValueError(f"Manifest entry lacks wsi_path: {entry}")
        item = dict(entry)
        item["wsi_path"] = str(resolve_path(item["wsi_path"], path.parent))
        if item.get("cached_h5_path"):
            item["cached_h5_path"] = str(resolve_path(item["cached_h5_path"], path.parent))
        if item.get("coordinate_h5_path"):
            item["coordinate_h5_path"] = str(resolve_path(item["coordinate_h5_path"], path.parent))
        item.setdefault("slide_id", Path(item["wsi_path"]).stem)
        resolved.append(item)
    return resolved


def filter_slides(slides: List[Dict[str, Any]], requested_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not requested_ids:
        return slides
    requested = set(requested_ids)
    filtered = [slide for slide in slides if slide["slide_id"] in requested]
    found = {slide["slide_id"] for slide in filtered}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Requested slide IDs not found in manifest: {missing}")
    return filtered


class PeakRssSampler:
    """Sample peak process-tree proportional set size (PSS)."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.peak_main_rss_bytes = 0
        self.peak_children_rss_bytes = 0
        self.peak_combined_rss_bytes = 0
        self.peak_combined_pss_bytes = 0
        self._last_pss_sample = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _snapshot(include_pss: bool) -> Dict[str, int]:
        process = psutil.Process(os.getpid())
        processes = [process]
        try:
            processes.extend(process.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        main_rss = 0
        children_rss = 0
        combined_pss = 0
        for proc in processes:
            try:
                rss = proc.memory_info().rss
                if proc.pid == process.pid:
                    main_rss = rss
                else:
                    children_rss += rss
                if include_pss:
                    try:
                        combined_pss += int(getattr(proc.memory_full_info(), "pss", 0))
                    except (psutil.Error, OSError, AttributeError):
                        pass
            except (psutil.Error, OSError):
                continue
        return {
            "main_rss": main_rss,
            "children_rss": children_rss,
            "combined_rss": main_rss + children_rss,
            "combined_pss": combined_pss,
        }

    def _sample_once(self, force_pss: bool = False) -> None:
        now = time.monotonic()
        include_pss = force_pss or now - self._last_pss_sample >= 1.0
        snapshot = self._snapshot(include_pss=include_pss)
        self.peak_main_rss_bytes = max(self.peak_main_rss_bytes, snapshot["main_rss"])
        self.peak_children_rss_bytes = max(self.peak_children_rss_bytes, snapshot["children_rss"])
        self.peak_combined_rss_bytes = max(self.peak_combined_rss_bytes, snapshot["combined_rss"])
        if include_pss:
            self.peak_combined_pss_bytes = max(self.peak_combined_pss_bytes, snapshot["combined_pss"])
            self._last_pss_sample = now
        self.peak_bytes = self.peak_combined_pss_bytes or self.peak_combined_rss_bytes

    def start(self) -> None:
        self.peak_bytes = 0
        self.peak_main_rss_bytes = 0
        self.peak_children_rss_bytes = 0
        self.peak_combined_rss_bytes = 0
        self.peak_combined_pss_bytes = 0
        self._last_pss_sample = 0.0
        self._sample_once(force_pss=True)
        self._stop.clear()

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._sample_once()

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample_once(force_pss=True)
        return self.peak_bytes

    def result_fields(self) -> Dict[str, int]:
        return {"peak_host_pss_bytes": self.peak_combined_pss_bytes}


class TimedLoader:
    """Record exposed DataLoader wait without counting prior GPU computation."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.wait_time = 0.0

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterable[torch.Tensor]:
        # Iterator creation can start workers and perform observable setup, so it
        # belongs to inference data loading. Synchronize before starting the
        # timer to exclude any outstanding GPU work from a preceding operation.
        synchronize()
        started = time.perf_counter()
        iterator = iter(self.loader)
        self.wait_time += time.perf_counter() - started
        while True:
            # Workers may prefetch while the preceding GPU forward is running.
            # The synchronization itself is intentionally outside the timed
            # interval; only the remaining critical-path wait for next() counts.
            synchronize()
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.wait_time += time.perf_counter() - started
            yield batch


def synchronize() -> None:
    torch.cuda.synchronize()


def failure_status(error: BaseException) -> str:
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
        return "oom"
    return "failed"


def configure_device(device_spec: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the released ModelDeployment implementation")
    device = torch.device(device_spec)
    if device.type != "cuda":
        raise ValueError("Only CUDA devices are supported")
    index = device.index if device.index is not None else 0
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def load_litefm_deployment(
    args: argparse.Namespace, device: torch.device
) -> Tuple[ModelDeployment, Any, float]:
    synchronize()
    started = time.perf_counter()
    deployment = ModelDeployment(
        args.model_name,
        n_classes=args.n_classes,
        k_a=args.k_a,
        k_u=args.k_u,
        aps_ckpt=str(Path(args.aps_ckpt).resolve()),
        mil_ckpt=str(Path(args.mil_ckpt).resolve()),
        buffer_threshold=args.buffer_threshold,
    )
    deployment.batch_size = args.batch_size
    deployment.litefm_model.eval()
    deployment.aps.eval()
    deployment.mil_model.eval()
    synchronize()
    elapsed = time.perf_counter() - started
    if deployment.device.index is not None and deployment.device.index != device.index:
        raise RuntimeError(f"Deployment loaded on {deployment.device}, expected {device}")
    transform = get_custom_transformer(args.model_name)
    return deployment, transform, elapsed


class FullPatchDeployment:
    """Minimal full-patch encoder + task-specific ABMIL inference interface."""

    def __init__(self, encoder: Any, mil_model: torch.nn.Module, device: torch.device, batch_size: int):
        self.encoder = encoder
        self.mil_model = mil_model
        self.device = device
        self.batch_size = batch_size

    def infer_feat_full(self, loader: TimedLoader) -> torch.Tensor:
        features = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for images in loader:
                images = images.to(self.device)
                features.append(self.encoder(images))
        # PathBench loads stored Virchow2 features as float32 before ABMIL.
        # Preserve that downstream dtype while keeping encoder computation in
        # FP16 autocast; the conversion remains inside the inference wall timer.
        return torch.cat(features, dim=0).float()


def get_virchow2_benchmark_transform() -> transforms.Compose:
    """Return the Virchow2 image transform used in the latency benchmark."""
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def load_virchow2_deployment(
    args: argparse.Namespace, device: torch.device
) -> Tuple[FullPatchDeployment, Any, float]:
    module_path = INFERENCE_ROOT.parent / "distillation" / "models" / "virchow2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Virchow2 implementation not found: {module_path}")
    spec = importlib.util.spec_from_file_location("litepath_benchmark_virchow2", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Virchow2 implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint_path = Path(args.virchow2_mil_ckpt).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Virchow2 ABMIL checkpoint not found: {checkpoint_path}")

    synchronize()
    started = time.perf_counter()
    transform = get_virchow2_benchmark_transform()
    encoder = module.get_virchow_model(device)
    mil_model = DAttention(
        n_classes=args.n_classes,
        dropout=0.25,
        act="relu",
        n_features=2560,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    first_weight = state_dict.get("feature.0.weight")
    classifier_weight = state_dict.get("classifier.0.weight")
    if first_weight is None or tuple(first_weight.shape) != (512, 2560):
        raise ValueError("Virchow2 ABMIL checkpoint does not have a 2560-dimensional input")
    if classifier_weight is None or classifier_weight.shape[0] != args.n_classes:
        raise ValueError("Virchow2 ABMIL checkpoint class count does not match --n-classes")
    mil_model.load_state_dict(state_dict, strict=True)
    mil_model.eval()
    synchronize()
    elapsed = time.perf_counter() - started
    return FullPatchDeployment(encoder, mil_model, device, args.batch_size), transform, elapsed


def warm_up_litefm(deployment: ModelDeployment, iterations: int, device: torch.device) -> float:
    if iterations <= 0:
        return 0.0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(iterations):
            images = torch.randn(8, 3, 224, 224, device=device)
            full_features = deployment.litefm_model(images)
            shallow = deployment.litefm_model.infer_deploy(images, stage="pre")
            aps_input = torch.cat([shallow[:, 0], shallow[:, 1:].mean(1)], dim=1)
            _ = deployment.aps(aps_input)
            _ = deployment.litefm_model.infer_deploy(shallow, stage="post")
            _ = deployment.mil_model(full_features)
    synchronize()
    return time.perf_counter() - started


def warm_up_full_patch(
    deployment: FullPatchDeployment, iterations: int, device: torch.device
) -> float:
    if iterations <= 0:
        return 0.0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(iterations):
            images = torch.randn(2, 3, 224, 224, device=device)
            features = deployment.encoder(images)
            _ = deployment.mil_model(features)
    synchronize()
    return time.perf_counter() - started


def build_loaders(
    packed_h5: Path,
    mode: str,
    transform: Any,
    args: argparse.Namespace,
) -> Tuple[Optional[TimedLoader], Optional[TimedLoader], Dict[str, int]]:
    dataset = PatchDataset(str(packed_h5), transform=transform, load_to_memory=False)
    total = len(dataset)
    if total <= 0:
        raise ValueError(f"Packed HDF5 contains no patches: {packed_h5}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    uniform_loader: Optional[TimedLoader] = None
    attention_loader: Optional[TimedLoader] = None
    uniform_count = 0
    attention_selected_count = 0
    attention_candidate_count = 0

    if mode == "aps":
        if total < args.k_u + args.k_a:
            raise ValueError(
                f"APS mode requires at least {args.k_u + args.k_a} patches; found {total}"
            )
        uniform_indices = torch.linspace(0, total - 1, steps=args.k_u).int()
        all_indices = torch.arange(total)
        remaining_indices = all_indices[~torch.isin(all_indices, uniform_indices)]
        uniform_loader = TimedLoader(DataLoader(Subset(dataset, uniform_indices), **loader_kwargs))
        attention_loader = TimedLoader(DataLoader(Subset(dataset, remaining_indices), **loader_kwargs))
        uniform_count = len(uniform_indices)
        attention_selected_count = args.k_a
        attention_candidate_count = len(remaining_indices)
    elif mode == "uniform":
        uniform_count = min(args.uniform_count, total)
        uniform_indices = torch.linspace(0, total - 1, steps=uniform_count).int()
        uniform_loader = TimedLoader(DataLoader(Subset(dataset, uniform_indices), **loader_kwargs))
    elif mode in {"litefm-full", "virchow2-full"}:
        uniform_count = total
        uniform_loader = TimedLoader(DataLoader(dataset, **loader_kwargs))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "aps":
        processed_patch_count = total
    else:
        processed_patch_count = uniform_count

    counts = {
        "tissue_patch_count": total,
        "processed_patch_count": processed_patch_count,
        "uniform_patch_count": uniform_count,
        "attention_candidate_count": attention_candidate_count,
        "attention_patch_count": attention_selected_count,
        "selected_patch_count": uniform_count + attention_selected_count,
    }
    return uniform_loader, attention_loader, counts


def run_cached_inference(
    deployment: Any,
    packed_h5: Path,
    mode: str,
    transform: Any,
    args: argparse.Namespace,
    rss_sampler: Optional[PeakRssSampler] = None,
) -> Dict[str, Any]:
    uniform_loader, attention_loader, counts = build_loaders(packed_h5, mode, transform, args)
    own_sampler = rss_sampler is None
    sampler = rss_sampler or PeakRssSampler(args.rss_interval)
    if own_sampler:
        sampler.start()

    torch.cuda.empty_cache()
    synchronize()
    torch.cuda.reset_peak_memory_stats()

    legacy_output = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(legacy_output):
            if attention_loader is None:
                # The released infer_litepath function assumes that an attention
                # loader exists and leaves attention_features uninitialized for a
                # uniform-only call. Reuse its existing feature and MIL methods
                # directly instead of changing the production implementation.
                uniform_features = deployment.infer_feat_full(uniform_loader)
                with torch.inference_mode():
                    logits = deployment.mil_model(uniform_features)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1)
            else:
                logits, probabilities, prediction = deployment.infer_litepath(
                    uniform_loader=uniform_loader,
                    attention_loader=attention_loader,
                )
        synchronize()
        logits_cpu = logits.detach().float().cpu()
        probabilities_cpu = probabilities.detach().float().cpu()
        prediction_cpu = prediction.detach().cpu()
        synchronize()
        cached_inference_wall_time = time.perf_counter() - started
    except Exception:
        if own_sampler:
            sampler.stop()
        raise

    peak_allocated = torch.cuda.max_memory_allocated()
    if own_sampler:
        sampler.stop()

    data_wait = 0.0
    if uniform_loader is not None:
        data_wait += uniform_loader.wait_time
    if attention_loader is not None:
        data_wait += attention_loader.wait_time

    inference_computation_time = cached_inference_wall_time - data_wait
    if inference_computation_time < 0:
        raise RuntimeError(
            "Measured DataLoader wait exceeds cached inference wall time; timing boundaries are invalid"
        )

    pred_index = int(prediction_cpu.reshape(-1)[0].item())
    result: Dict[str, Any] = {
        **counts,
        "data_loading_wait_time": data_wait,
        "inference_computation_time": inference_computation_time,
        "cached_inference_wall_time": cached_inference_wall_time,
        "peak_gpu_memory_bytes": peak_allocated,
        **sampler.result_fields(),
        "logits": logits_cpu.numpy().tolist(),
        "probabilities": probabilities_cpu.numpy().tolist(),
        "prediction_index": pred_index,
        "prediction_label": args.labels[pred_index] if pred_index < len(args.labels) else str(pred_index),
    }

    del uniform_loader, attention_loader, logits, probabilities, prediction
    return result


def parse_identifier_list(value: Any) -> List[int]:
    text = str(value)
    if text.lower() in {"none", "", "nan"}:
        return []
    return [int(item) for item in text.split(",")]


def load_tcga_preset(path: Path) -> Dict[str, Dict[str, Any]]:
    row = pd.read_csv(path).iloc[0]
    return {
        "seg_params": {
            "seg_level": int(row["seg_level"]),
            "sthresh": int(row["sthresh"]),
            "mthresh": int(row["mthresh"]),
            "close": int(row["close"]),
            "use_otsu": bool(row["use_otsu"]),
            "keep_ids": parse_identifier_list(row["keep_ids"]),
            "exclude_ids": parse_identifier_list(row["exclude_ids"]),
        },
        "filter_params": {
            "a_t": float(row["a_t"]),
            "a_h": float(row["a_h"]),
            "max_n_holes": int(row["max_n_holes"]),
        },
        "patch_params": {
            "use_padding": bool(row["use_padding"]),
            "contour_fn": str(row["contour_fn"]),
        },
    }


def extract_coordinates(wsi_path: Path, coord_h5: Path, preset_path: Path) -> Dict[str, Any]:
    coord_h5.parent.mkdir(parents=True, exist_ok=True)
    if coord_h5.exists():
        raise FileExistsError(coord_h5)
    preset = load_tcga_preset(preset_path)
    coordinate_started = time.perf_counter()
    wsi_object = WholeSlideImage(str(wsi_path))

    mpp = wsi_object.mpp
    if mpp is None:
        mpp = 0.25
        object_power = 40
        mpp_fallback = True
    else:
        object_power = int(round(10.0 / float(mpp)))
        mpp_fallback = False
    patch_size, step_size = adjust_size(object_power)

    seg_params = dict(preset["seg_params"])
    if seg_params["seg_level"] < 0:
        if len(wsi_object.level_dim) == 1:
            seg_params["seg_level"] = 0
        else:
            seg_params["seg_level"] = wsi_object.getOpenSlide().get_best_level_for_downsample(
                estimate_best_seg_level(wsi_object)
            )

    wsi_object.segmentTissue(**seg_params, filter_params=preset["filter_params"])
    if not wsi_object.contours_tissue:
        raise RuntimeError(f"No tissue contours found for {wsi_path}")

    wsi_object.process_contours(
        save_path=str(coord_h5.parent),
        patch_level=0,
        patch_size=patch_size,
        step_size=step_size,
        **preset["patch_params"],
    )

    if not coord_h5.exists():
        raise RuntimeError(f"Coordinate generation did not create {coord_h5}")
    with h5py.File(coord_h5, "r") as handle:
        tissue_patch_count = int(len(handle["coords"]))

    close_method = getattr(wsi_object.wsi, "close", None)
    if callable(close_method):
        close_method()
    del wsi_object
    coordinate_extraction_time = time.perf_counter() - coordinate_started
    return {
        "coordinate_extraction_time": coordinate_extraction_time,
        "tissue_patch_count": tissue_patch_count,
        "mpp": float(mpp),
        "mpp_fallback": mpp_fallback,
        "object_power": object_power,
        "segmentation_level": int(seg_params["seg_level"]),
        "patch_level": 0,
        "patch_size": patch_size,
        "step_size": step_size,
        "preset_path": str(preset_path.resolve()),
    }


def materialize_hdf5(wsi_path: Path, coord_h5: Path, packed_h5: Path) -> Dict[str, Any]:
    packed_h5.parent.mkdir(parents=True, exist_ok=True)
    if packed_h5.exists():
        raise FileExistsError(packed_h5)
    with h5py.File(coord_h5, "r") as handle:
        tissue_patch_count = int(len(handle["coords"]))
    materialization_started = time.perf_counter()
    read_images((str(coord_h5), str(packed_h5), str(wsi_path)))
    if not packed_h5.exists():
        raise RuntimeError(f"HDF5 materialization did not create {packed_h5}")
    with h5py.File(packed_h5, "r") as handle:
        packed_patch_count = int(len(handle["patches"]))
    if packed_patch_count != tissue_patch_count:
        raise RuntimeError(
            f"Coordinate/packed patch mismatch: {tissue_patch_count} vs {packed_patch_count}"
        )
    hdf5_materialization_time = time.perf_counter() - materialization_started
    return {
        "hdf5_materialization_time": hdf5_materialization_time,
        "tissue_patch_count": tissue_patch_count,
    }


def inspect_cache(coord_h5: Path, packed_h5: Path) -> Dict[str, Any]:
    with h5py.File(coord_h5, "r") as handle:
        coordinate_count = int(len(handle["coords"]))
    with h5py.File(packed_h5, "r") as handle:
        packed_count = int(len(handle["patches"]))
    if coordinate_count != packed_count:
        raise RuntimeError(
            f"Coordinate/packed patch mismatch: {coordinate_count} vs {packed_count}"
        )
    return {
        "tissue_patch_count": coordinate_count,
        "coordinate_h5_bytes": coord_h5.stat().st_size,
        "packed_h5_bytes": packed_h5.stat().st_size,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def collect_environment(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    environment = {
        "collected_at_utc": utc_now(),
        "hardware_label": args.hardware_label or props.name,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": run_command([sys.executable, "-c", "import torchvision; print(torchvision.__version__)"]),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_spec": str(device),
        "device_name": props.name,
        "device_total_memory_bytes": props.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "h5py": h5py.__version__,
        "pillow": run_command([sys.executable, "-c", "import PIL; print(PIL.__version__)"]),
        "psutil": psutil.__version__,
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short"]),
        "nvidia_smi": run_command(["nvidia-smi", "-q"], timeout=30),
        "jetson_power_mode": run_command(["nvpmodel", "-q"]),
        "jetson_clocks": run_command(["jetson_clocks", "--show"]),
    }
    return environment


def collect_preprocessing_environment() -> Dict[str, Any]:
    return {
        "collected_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "h5py": h5py.__version__,
        "pillow": run_command([sys.executable, "-c", "import PIL; print(PIL.__version__)"]),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short"]),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    preferred = [
        "status", "slide_id", "task", "mode", "method", "encoder",
        "all_patches", "repeat", "tissue_patch_count", "processed_patch_count",
        "selected_patch_count", "attention_patch_count",
        "coordinate_extraction_time", "hdf5_materialization_time",
        "coordinate_timing_source", "materialization_timing_source",
        "coordinate_reused", "packed_hdf5_reused", "packed_hdf5_size",
        "data_loading_wait_time", "inference_computation_time",
        "cached_inference_wall_time",
        "peak_gpu_memory_bytes", "peak_host_pss_bytes",
        "intermediate_hdf5_storage_bytes",
        "hardware", "batch_size", "num_workers", "pin_memory", "prefetch_factor",
        "precision", "cache_mode",
    ]
    if not rows:
        return
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in fieldnames:
                value = row.get(key)
                flat[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            writer.writerow(flat)
    temp.replace(path)


LATENCY_METRICS = [
    "data_loading_wait_time",
    "inference_computation_time",
    "cached_inference_wall_time",
]

RESOURCE_METRICS = [
    "peak_gpu_memory_bytes",
    "peak_host_pss_bytes",
    "intermediate_hdf5_storage_bytes",
]

PREPROCESS_METRICS = [
    "coordinate_extraction_time",
    "hdf5_materialization_time",
]


def summarize(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            key = (
                str(row.get("slide_id")),
                str(row.get("task")),
                str(row.get("mode")),
            )
            groups[key].append(row)

    output: List[Dict[str, Any]] = []
    for (slide_id, task, mode), items in groups.items():
        summary: Dict[str, Any] = {
            "slide_id": slide_id,
            "task": task,
            "mode": mode,
            "method": items[0].get("method"),
            "encoder": items[0].get("encoder"),
            "all_patches": items[0].get("all_patches"),
            "n_success": len(items),
            "tissue_patch_count": items[0].get("tissue_patch_count"),
            "processed_patch_count": items[0].get("processed_patch_count"),
            "selected_patch_count": items[0].get("selected_patch_count"),
            "hardware": items[0].get("hardware"),
            "batch_size": items[0].get("batch_size"),
            "num_workers": items[0].get("num_workers"),
            "pin_memory": items[0].get("pin_memory"),
            "prefetch_factor": items[0].get("prefetch_factor"),
            "precision": items[0].get("precision"),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def build_sanity_checks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in successful:
        groups[(str(row.get("slide_id")), str(row.get("mode")))].append(row)

    group_checks = []
    all_passed = not failures
    for (slide_id, mode), items in sorted(groups.items()):
        inference_partition_errors = [
            abs(
                float(item.get("cached_inference_wall_time", 0.0))
                - float(item.get("data_loading_wait_time", 0.0))
                - float(item.get("inference_computation_time", 0.0))
            )
            for item in items
        ]
        predictions = sorted({str(item.get("prediction_label")) for item in items})
        patch_counts = sorted({int(item["tissue_patch_count"]) for item in items})
        processed_counts = sorted({int(item["processed_patch_count"]) for item in items})
        full_patch_checks = [
            int(item["processed_patch_count"]) == int(item["tissue_patch_count"])
            for item in items
            if bool(item.get("all_patches"))
        ]
        passed = (
            max(inference_partition_errors, default=0.0) < 1e-9
            and len(predictions) == 1
            and len(patch_counts) == 1
            and len(processed_counts) == 1
            and all(full_patch_checks)
        )
        all_passed = all_passed and passed
        group_checks.append(
            {
                "slide_id": slide_id,
                "mode": mode,
                "n_runs": len(items),
                "predictions": predictions,
                "tissue_patch_counts": patch_counts,
                "processed_patch_counts": processed_counts,
                "full_patch_count_matches_tissue_count": (
                    all(full_patch_checks) if full_patch_checks else None
                ),
                "max_inference_partition_absolute_error_seconds": max(
                    inference_partition_errors, default=0.0
                ),
                "passed": passed,
            }
        )
    return {
        "updated_at_utc": utc_now(),
        "all_passed": all_passed,
        "n_success": len(successful),
        "n_failures": len(failures),
        "failures": failures,
        "groups": group_checks,
    }


def derive_end_to_end_summary(
    inference_summary: List[Dict[str, Any]], slides: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    slide_lookup = {str(slide["slide_id"]): slide for slide in slides}
    output = []
    for row in inference_summary:
        slide = slide_lookup[str(row["slide_id"])]
        coordinate_time = slide.get("coordinate_extraction_time")
        materialization_time = slide.get("hdf5_materialization_time")
        inference_time = row.get("cached_inference_wall_time_mean")
        derived = dict(row)
        derived["coordinate_extraction_time"] = coordinate_time
        derived["hdf5_materialization_time"] = materialization_time
        if coordinate_time is not None and materialization_time is not None and inference_time is not None:
            derived["derived_raw_wsi_to_prediction_time"] = (
                float(coordinate_time) + float(materialization_time) + float(inference_time)
            )
        else:
            derived["derived_raw_wsi_to_prediction_time"] = None
        output.append(derived)
    return output


def persist_inference_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    environment: Dict[str, Any],
    run_config: Dict[str, Any],
    slides: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_summary = summarize(rows, LATENCY_METRICS)
    write_csv(output_dir / "inference_results.csv", rows)
    write_csv(output_dir / "inference_summary.csv", inference_summary)
    write_csv(output_dir / "resource_summary.csv", summarize(rows, RESOURCE_METRICS))
    write_csv(
        output_dir / "derived_end_to_end_summary.csv",
        derive_end_to_end_summary(inference_summary, slides),
    )
    (output_dir / "sanity_checks.json").write_text(
        json.dumps(json_ready(build_sanity_checks(rows)), indent=2), encoding="utf-8"
    )
    (output_dir / "inference_environment.json").write_text(
        json.dumps(json_ready(environment), indent=2), encoding="utf-8"
    )
    (output_dir / "inference_config.json").write_text(
        json.dumps(json_ready(run_config), indent=2), encoding="utf-8"
    )


def common_result(
    slide: Dict[str, Any],
    args: argparse.Namespace,
    mode: str,
    repeat: int,
) -> Dict[str, Any]:
    wsi_path = Path(slide["wsi_path"])
    metadata = mode_metadata(mode)
    return {
        "status": "running",
        "started_at_utc": utc_now(),
        "slide_id": slide["slide_id"],
        "task": args.task,
        "mode": mode,
        **metadata,
        "repeat": repeat,
        "wsi_path": str(wsi_path),
        "raw_wsi_bytes": wsi_path.stat().st_size if wsi_path.exists() else None,
        "hardware": args.hardware_label or torch.cuda.get_device_name(),
        "device": str(torch.cuda.current_device()),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "precision": "fp16_autocast",
        "cache_mode": "prepared_hdf5",
        "packed_h5_sha256": slide.get("packed_h5_sha256"),
    }


def validate_prepared_cache(slide: Dict[str, Any]) -> Dict[str, Any]:
    coordinate_value = slide.get("coordinate_h5_path")
    packed_value = slide.get("cached_h5_path")
    if not packed_value:
        raise ValueError(f"Prepared manifest lacks cached_h5_path for {slide['slide_id']}")
    packed_h5 = Path(packed_value)
    if not packed_h5.exists():
        raise FileNotFoundError(packed_h5)
    with h5py.File(packed_h5, "r") as handle:
        packed_count = int(len(handle["patches"]))
    inspected = {
        "tissue_patch_count": packed_count,
        "packed_h5_bytes": packed_h5.stat().st_size,
    }
    expected_count = slide.get("tissue_patch_count")
    if expected_count is not None and int(expected_count) != inspected["tissue_patch_count"]:
        raise RuntimeError(f"Prepared cache patch count changed for {slide['slide_id']}")
    packed_hash = sha256_file(packed_h5)
    if slide.get("packed_h5_sha256") and slide["packed_h5_sha256"] != packed_hash:
        raise RuntimeError(f"Packed HDF5 hash mismatch for {slide['slide_id']}")
    inspected["packed_h5_sha256"] = packed_hash
    if coordinate_value and Path(coordinate_value).exists():
        coordinate_h5 = Path(coordinate_value)
        with h5py.File(coordinate_h5, "r") as handle:
            coordinate_count = int(len(handle["coords"]))
        if coordinate_count != packed_count:
            raise RuntimeError(
                f"Coordinate/packed patch mismatch: {coordinate_count} vs {packed_count}"
            )
        coordinate_hash = sha256_file(coordinate_h5)
        if slide.get("coordinate_h5_sha256") and slide["coordinate_h5_sha256"] != coordinate_hash:
            LOGGER.warning(
                "Coordinate HDF5 hash mismatch for %s; inference uses packed HDF5 only, continuing",
                slide["slide_id"],
            )
        inspected.update(
            {
                "coordinate_h5_bytes": coordinate_h5.stat().st_size,
                "coordinate_h5_sha256": coordinate_hash,
            }
        )
    return inspected


def run_inference_workflows(
    slides: List[Dict[str, Any]],
    deployment: ModelDeployment,
    transform: Any,
    args: argparse.Namespace,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    environment: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    for slide in slides:
        packed_h5_value = slide.get("cached_h5_path")
        if not packed_h5_value:
            raise ValueError(f"Prepared manifest lacks cached_h5_path for {slide['slide_id']}")
        packed_h5 = Path(packed_h5_value)
        if not packed_h5.exists():
            raise FileNotFoundError(packed_h5)
        for mode in args.mode:
            for repeat in range(1, args.repeats + 1):
                result = common_result(slide, args, mode, repeat)
                result.update(
                    {
                        "packed_h5_path": str(packed_h5.resolve()),
                        "packed_h5_bytes": packed_h5.stat().st_size,
                        "intermediate_hdf5_storage_bytes": packed_h5.stat().st_size,
                    }
                )
                coordinate_h5_value = slide.get("coordinate_h5_path")
                if coordinate_h5_value:
                    coordinate_h5 = Path(coordinate_h5_value)
                    if coordinate_h5.exists():
                        result["coordinate_h5_path"] = str(coordinate_h5.resolve())
                        result["coordinate_h5_bytes"] = coordinate_h5.stat().st_size
                        result["intermediate_hdf5_storage_bytes"] += coordinate_h5.stat().st_size
                try:
                    LOGGER.info("Inference: slide=%s mode=%s repeat=%d", slide["slide_id"], mode, repeat)
                    inference = run_cached_inference(deployment, packed_h5, mode, transform, args)
                    result.update(inference)
                    gc.collect()
                    torch.cuda.empty_cache()
                    result["status"] = "success"
                    result["finished_at_utc"] = utc_now()
                    LOGGER.info(
                        "Completed cached workflow: slide=%s mode=%s repeat=%d total=%.3fs prediction=%s",
                        slide["slide_id"], mode, repeat, result["cached_inference_wall_time"], result["prediction_label"],
                    )
                except Exception as error:
                    result["status"] = failure_status(error)
                    result["failure_stage"] = "cached_inference"
                    result["error"] = repr(error)
                    result["traceback"] = traceback.format_exc()
                    result["finished_at_utc"] = utc_now()
                    LOGGER.exception("Cached workflow failed for %s mode %s", slide["slide_id"], mode)
                    gc.collect()
                    torch.cuda.empty_cache()
                rows.append(json_ready(result))
                persist_inference_outputs(output_dir, rows, environment, run_config, slides)
                if result["status"] != "success" and args.fail_fast:
                    raise RuntimeError(result["error"])


def relative_path(path: Path, base_dir: Path) -> str:
    # Keep the cache layout encoded in the manifest.  Resolving here would
    # dereference symbolic links and could make an otherwise portable manifest
    # point back to an unrelated source cache.
    return os.path.relpath(os.path.abspath(path), os.path.abspath(base_dir))


def write_preprocessing_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    prepared_slides: List[Dict[str, Any]],
    environment: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "preprocessing_results.csv", rows)
    write_csv(output_dir / "preprocessing_summary.csv", summarize(rows, PREPROCESS_METRICS))
    atomic_write_json(output_dir / "prepared_cache_manifest.json", {"slides": prepared_slides})
    atomic_write_json(output_dir / "preprocessing_environment.json", environment)
    atomic_write_json(output_dir / "preprocessing_config.json", config)


def run_preprocessing(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "preprocessing.log", encoding="utf-8"),
        ],
    )

    manifest_path = Path(args.manifest).expanduser().resolve()
    slides = filter_slides(load_manifest(manifest_path), args.slide_id)
    preset_path = Path(args.preset).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    prepared_slides: List[Dict[str, Any]] = []
    environment = collect_preprocessing_environment()
    config = {
        "created_at_utc": utc_now(),
        "command": "preprocess",
        "arguments": vars(args),
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "timing_notes": {
            "coordinate_extraction_time_is_continuous_wall_time": True,
            "hdf5_materialization_time_is_continuous_wall_time": True,
            "hashing_and_cache_validation_excluded_from_timers": True,
            "preprocessing_has_no_cuda_or_model_dependency": True,
        },
    }

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    for slide in slides:
        for repeat in range(1, args.repeats + 1):
            slide_id = str(slide["slide_id"])
            wsi_path = Path(slide["wsi_path"])
            coord_h5 = cache_dir / "coords_h5" / f"{slide_id}.h5"
            packed_h5 = cache_dir / "packed_images" / f"{slide_id}.h5"
            metadata_path = cache_dir / "metadata" / f"{slide_id}.json"
            row: Dict[str, Any] = {
                "status": "running",
                "slide_id": slide_id,
                "mode": "preprocess",
                "repeat": repeat,
                "wsi_path": str(wsi_path),
                "coordinate_h5_path": str(coord_h5),
                "cached_h5_path": str(packed_h5),
                "coordinate_extraction_time": None,
                "hdf5_materialization_time": None,
                "coordinate_timing_source": "unavailable",
                "materialization_timing_source": "unavailable",
                "started_at_utc": utc_now(),
            }
            try:
                if not wsi_path.exists():
                    raise FileNotFoundError(wsi_path)
                previous: Dict[str, Any] = {}
                if metadata_path.exists():
                    previous = json.loads(metadata_path.read_text(encoding="utf-8"))

                if packed_h5.exists() and not coord_h5.exists() and not args.force_preprocess and args.repeats == 1:
                    raise RuntimeError(
                        f"Packed HDF5 exists without its coordinate HDF5 for {slide_id}"
                    )

                measure = args.force_preprocess or args.repeats > 1
                regenerate_coordinates = measure or not coord_h5.exists()
                regenerate_packed = measure or not packed_h5.exists()
                coordinate_info: Dict[str, Any] = {}
                materialization_info: Dict[str, Any] = {}

                with tempfile.TemporaryDirectory(prefix=f".{slide_id}.", dir=cache_dir) as temporary:
                    temporary_root = Path(temporary)
                    working_coord = coord_h5
                    if regenerate_coordinates:
                        working_coord = temporary_root / "coords_h5" / f"{wsi_path.stem}.h5"
                        coordinate_info = extract_coordinates(wsi_path, working_coord, preset_path)
                        row["coordinate_extraction_time"] = coordinate_info["coordinate_extraction_time"]
                        row["coordinate_timing_source"] = "measured"
                    else:
                        with h5py.File(coord_h5, "r") as handle:
                            coordinate_info["tissue_patch_count"] = int(len(handle["coords"]))
                        row["coordinate_extraction_time"] = previous.get("coordinate_extraction_time")
                        row["coordinate_timing_source"] = (
                            "reused_record" if row["coordinate_extraction_time"] is not None else "unavailable"
                        )

                    working_packed = packed_h5
                    if regenerate_packed:
                        working_packed = temporary_root / "packed_images" / f"{slide_id}.h5"
                        materialization_info = materialize_hdf5(
                            wsi_path, working_coord, working_packed
                        )
                        row["hdf5_materialization_time"] = materialization_info[
                            "hdf5_materialization_time"
                        ]
                        row["materialization_timing_source"] = "measured"
                    else:
                        row["hdf5_materialization_time"] = previous.get(
                            "hdf5_materialization_time"
                        )
                        row["materialization_timing_source"] = (
                            "reused_record"
                            if row["hdf5_materialization_time"] is not None
                            else "unavailable"
                        )

                    inspect_cache(working_coord, working_packed)
                    if regenerate_coordinates:
                        coord_h5.parent.mkdir(parents=True, exist_ok=True)
                        working_coord.replace(coord_h5)
                    if regenerate_packed:
                        packed_h5.parent.mkdir(parents=True, exist_ok=True)
                        working_packed.replace(packed_h5)

                inspected = inspect_cache(coord_h5, packed_h5)
                coordinate_hash = sha256_file(coord_h5)
                packed_hash = sha256_file(packed_h5)
                if not regenerate_coordinates:
                    if previous.get("coordinate_h5_sha256") not in {None, coordinate_hash}:
                        raise RuntimeError(f"Coordinate HDF5 hash changed for {slide_id}")
                if not regenerate_packed:
                    if previous.get("packed_h5_sha256") not in {None, packed_hash}:
                        raise RuntimeError(f"Packed HDF5 hash changed for {slide_id}")

                row.update(
                    {
                        **coordinate_info,
                        **materialization_info,
                        **inspected,
                        "packed_hdf5_size": inspected["packed_h5_bytes"],
                        "coordinate_h5_sha256": coordinate_hash,
                        "packed_h5_sha256": packed_hash,
                        "cache_reused": not regenerate_coordinates and not regenerate_packed,
                        "coordinate_reused": not regenerate_coordinates,
                        "packed_hdf5_reused": not regenerate_packed,
                        "status": "success",
                        "finished_at_utc": utc_now(),
                    }
                )
                # Restore timing fields after merging stage metadata.
                if not regenerate_coordinates:
                    row["coordinate_extraction_time"] = previous.get("coordinate_extraction_time")
                if not regenerate_packed:
                    row["hdf5_materialization_time"] = previous.get("hdf5_materialization_time")
                atomic_write_json(metadata_path, row)
                prepared_slides[:] = [item for item in prepared_slides if item["slide_id"] != slide_id]
                prepared_slides.append(
                    {
                        "slide_id": slide_id,
                        "wsi_path": relative_path(wsi_path, output_dir),
                        "coordinate_h5_path": relative_path(coord_h5, output_dir),
                        "cached_h5_path": relative_path(packed_h5, output_dir),
                        "tissue_patch_count": inspected["tissue_patch_count"],
                        "coordinate_h5_bytes": inspected["coordinate_h5_bytes"],
                        "packed_h5_bytes": inspected["packed_h5_bytes"],
                        "packed_hdf5_size": inspected["packed_h5_bytes"],
                        "coordinate_h5_sha256": coordinate_hash,
                        "packed_h5_sha256": packed_hash,
                        "coordinate_extraction_time": row.get("coordinate_extraction_time"),
                        "hdf5_materialization_time": row.get("hdf5_materialization_time"),
                        "coordinate_timing_source": row["coordinate_timing_source"],
                        "materialization_timing_source": row["materialization_timing_source"],
                    }
                )
                LOGGER.info(
                    "Prepared slide=%s repeat=%d patches=%d coordinate=%s materialization=%s",
                    slide_id,
                    repeat,
                    inspected["tissue_patch_count"],
                    row["coordinate_timing_source"],
                    row["materialization_timing_source"],
                )
            except Exception as error:
                row.update(
                    {
                        "status": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "finished_at_utc": utc_now(),
                    }
                )
                LOGGER.exception("Preprocessing failed for %s repeat %d", slide_id, repeat)
            rows.append(json_ready(row))
            write_preprocessing_outputs(output_dir, rows, prepared_slides, environment, config)
            if row["status"] != "success" and args.fail_fast:
                raise RuntimeError(row["error"])

    return 1 if any(row["status"] != "success" for row in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess", help="Create or reuse coordinate and packed-patch HDF5 caches"
    )
    preprocess.add_argument("--manifest", required=True, help="JSON manifest containing raw WSI paths")
    preprocess.add_argument("--slide-id", action="append")
    preprocess.add_argument("--cache-dir", required=True)
    preprocess.add_argument("--output-dir", required=True)
    preprocess.add_argument("--preset", default="presets/tcga.csv")
    preprocess.add_argument("--repeats", type=int, default=1)
    preprocess.add_argument("--force-preprocess", action="store_true")
    preprocess.add_argument("--fail-fast", action="store_true")

    inference = subparsers.add_parser(
        "inference", help="Benchmark inference from an existing prepared-cache manifest"
    )
    inference.add_argument("--manifest", required=True, help="Prepared-cache manifest")
    inference.add_argument("--slide-id", action="append")
    inference.add_argument(
        "--method",
        action="append",
        choices=["aps", "uniform", "litefm", "virchow2"],
        help="Repeat to benchmark multiple methods (default: aps)",
    )
    inference.add_argument("--repeats", type=int, default=3)
    inference.add_argument("--output-dir", required=True)
    inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--hardware-label", default=None)
    inference.add_argument(
        "--hardware-preset",
        choices=["rtx3090", "jetson_orin_nano_super"],
        default="rtx3090",
    )
    inference.add_argument("--batch-size", type=int, default=None, help="LiteFM batch size")
    inference.add_argument("--virchow2-batch-size", type=int, default=None)
    inference.add_argument("--num-workers", type=int, default=None)
    pin_memory = inference.add_mutually_exclusive_group()
    pin_memory.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    pin_memory.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    inference.set_defaults(pin_memory=None)
    inference.add_argument("--prefetch-factor", type=int, default=2)
    inference.add_argument("--warmup-iterations", type=int, default=5)
    inference.add_argument("--rss-interval", type=float, default=0.05)
    inference.add_argument("--seed", type=int, default=0)
    inference.add_argument("--task", default="NSCLC")
    inference.add_argument("--model-name", default="LiteFM")
    inference.add_argument("--n-classes", type=int, default=2)
    inference.add_argument("--labels", nargs="+", default=["LUAD", "LUSC"])
    inference.add_argument("--k-u", type=int, default=1900)
    inference.add_argument("--k-a", type=int, default=100)
    inference.add_argument("--uniform-count", type=int, default=2000)
    inference.add_argument("--buffer-threshold", type=int, default=2500)
    inference.add_argument("--aps-ckpt", default="examples/NSCLC/models/aps_model_best.pth.tar")
    inference.add_argument("--mil-ckpt", default="examples/NSCLC/models/model_best.pth.tar")
    inference.add_argument(
        "--virchow2-mil-ckpt",
        default="examples/NSCLC/models/virchow2_model_best.pth.tar",
        help="Task-specific 2560-dimensional Virchow2 ABMIL checkpoint",
    )
    inference.add_argument("--fail-fast", action="store_true")
    return parser


def validate_inference_args(args: argparse.Namespace) -> None:
    if args.hardware_preset == "jetson_orin_nano_super":
        args.batch_size = 256 if args.batch_size is None else args.batch_size
        args.virchow2_batch_size = 1 if args.virchow2_batch_size is None else args.virchow2_batch_size
        args.num_workers = 6 if args.num_workers is None else args.num_workers
        args.pin_memory = False if args.pin_memory is None else args.pin_memory
        args.hardware_label = args.hardware_label or "Jetson_Orin_Nano_Super"
    else:
        args.batch_size = 256 if args.batch_size is None else args.batch_size
        args.virchow2_batch_size = 16 if args.virchow2_batch_size is None else args.virchow2_batch_size
        args.num_workers = 32 if args.num_workers is None else args.num_workers
        args.pin_memory = True if args.pin_memory is None else args.pin_memory
        args.hardware_label = args.hardware_label or "RTX_3090"
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if (
        args.batch_size < 1
        or args.virchow2_batch_size < 1
        or args.num_workers < 0
        or args.prefetch_factor < 1
    ):
        raise ValueError("Invalid batch size or worker count")
    if args.k_u < 0 or args.k_a < 0 or args.uniform_count < 1:
        raise ValueError("Invalid patch selection counts")
    if args.k_a > args.buffer_threshold:
        raise ValueError("k_a must not exceed buffer_threshold")
    method_to_mode = {
        "aps": "aps",
        "uniform": "uniform",
        "litefm": "litefm-full",
        "virchow2": "virchow2-full",
    }
    requested_methods = args.method or ["aps"]
    args.mode = list(dict.fromkeys(method_to_mode[method] for method in requested_methods))


def run_inference_benchmark(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "inference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )

    manifest_path = Path(args.manifest).expanduser().resolve()
    slides = filter_slides(load_manifest(manifest_path), args.slide_id)
    for slide in slides:
        slide.update(validate_prepared_cache(slide))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = configure_device(args.device)

    environment = collect_environment(args, device)
    run_config = {
        "created_at_utc": utc_now(),
        "command": "inference",
        "arguments": vars(args),
        "manifest_path": str(manifest_path),
        "slides": slides,
        "timing_notes": {
            "model_loading_excluded_from_per_slide_latency": True,
            "preprocessing_excluded_from_inference": True,
            "cache_hash_validation_excluded_from_inference_timers": True,
            "data_loading_definition": (
                "DataLoader iterator initialization plus cumulative successful next() wait; CUDA is "
                "synchronized immediately before each wait timer so prior asynchronous computation is excluded"
            ),
            "inference_computation_definition": (
                "cached_inference_wall_time minus data_loading_wait_time"
            ),
            "inference_computation_includes": [
                "H2D transfer", "model forwards", "buffer operations", "concatenation", "mean",
                "top-k", "softmax", "argmax", "Python orchestration", "CUDA synchronization"
            ],
            "elapsed_times_directly_measured": True,
            "os_page_cache_flushed": False,
            "network_transfer_included": False,
            "mask_generation_included": False,
            "stitching_included": False,
        },
    }

    rows: List[Dict[str, Any]] = []
    environment["models"] = {}
    persist_inference_outputs(output_dir, rows, environment, run_config, slides)

    mode_groups = [
        ("LiteFM", [mode for mode in args.mode if mode_metadata(mode)["encoder"] == "LiteFM"]),
        (
            "Virchow2",
            [mode for mode in args.mode if mode_metadata(mode)["encoder"] == "Virchow2"],
        ),
    ]
    for encoder_name, modes in mode_groups:
        if not modes:
            continue
        group_args = copy.copy(args)
        group_args.mode = modes
        if encoder_name == "Virchow2":
            group_args.batch_size = args.virchow2_batch_size

        LOGGER.info(
            "Loading %s models on %s for modes=%s batch_size=%d",
            encoder_name,
            device,
            modes,
            group_args.batch_size,
        )
        deployment = None
        transform = None
        try:
            if encoder_name == "LiteFM":
                deployment, transform, model_load_time = load_litefm_deployment(group_args, device)
                warmup_time = warm_up_litefm(deployment, args.warmup_iterations, device)
            else:
                deployment, transform, model_load_time = load_virchow2_deployment(group_args, device)
                warmup_time = warm_up_full_patch(deployment, args.warmup_iterations, device)
            environment["models"][encoder_name] = {
                "model_load_time": model_load_time,
                "warmup_time": warmup_time,
                "post_warmup_gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "post_warmup_gpu_reserved_bytes": torch.cuda.memory_reserved(),
                "batch_size": group_args.batch_size,
            }
            persist_inference_outputs(output_dir, rows, environment, run_config, slides)
            LOGGER.info(
                "%s load %.3fs; warm-up %.3fs",
                encoder_name,
                model_load_time,
                warmup_time,
            )
        except Exception as error:
            status = failure_status(error)
            LOGGER.exception("Could not load or warm up encoder group %s", encoder_name)
            environment["models"][encoder_name] = {
                "status": status,
                "error": repr(error),
                "batch_size": group_args.batch_size,
            }
            for slide in slides:
                for mode in modes:
                    for repeat in range(1, args.repeats + 1):
                        failure = common_result(slide, group_args, mode, repeat)
                        failure.update(
                            {
                                "status": status,
                                "failure_stage": "model_loading_or_warmup",
                                "error": repr(error),
                                "traceback": traceback.format_exc(),
                                "finished_at_utc": utc_now(),
                            }
                        )
                        rows.append(json_ready(failure))
            persist_inference_outputs(output_dir, rows, environment, run_config, slides)
            if args.fail_fast:
                raise
            if deployment is not None:
                del deployment
            if transform is not None:
                del transform
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            run_inference_workflows(
                slides,
                deployment,
                transform,
                group_args,
                output_dir,
                rows,
                environment,
                run_config,
            )
        finally:
            del deployment, transform
            gc.collect()
            torch.cuda.empty_cache()

    failures = [row for row in rows if row.get("status") != "success"]
    LOGGER.info("Benchmark complete: %d runs, %d failures", len(rows), len(failures))
    return 1 if failures else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "preprocess":
        return run_preprocessing(args)
    validate_inference_args(args)
    return run_inference_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
