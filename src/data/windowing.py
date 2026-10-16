from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.lerobot_reader import group_episode_frames, sort_frames, stack_feature_matrix

DEFAULT_FEATURES = ("observation.state", "action")


def slice_episode(values: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    """Cut one episode's [T, F] matrix into fixed windows (no boundary crossing)."""
    if len(values) < window_size:
        return []
    return [values[start : start + window_size] for start in range(0, len(values) - window_size + 1, stride)]


def make_sliding_windows(
    df: pd.DataFrame,
    features: list[str] | tuple[str, ...] = DEFAULT_FEATURES,
    window_size: int = 60,
    stride: int = 30,
) -> np.ndarray:
    """Slice a single dataset's frames into fixed windows.

    Windows never cross episode boundaries: frames are grouped by ``episode_index``
    first, then sliced per episode. ``features`` are concatenated column-wise
    (observation.state + action -> 12D).
    """
    if isinstance(features, str):
        features = (features,)
    feature_dim = 0
    windows: list[np.ndarray] = []
    for _, episode in group_episode_frames(sort_frames(df)):
        values = stack_feature_matrix(episode, features)
        feature_dim = values.shape[1]
        windows.extend(slice_episode(values, window_size, stride))
    if not windows:
        return np.empty((0, window_size, feature_dim), dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


def normalization_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (per-joint/per-channel) mean and std, shape [1, 1, F].

    Computed on a single dataset only (per-dataset normalization): pretraining uses
    the external stats, fine-tuning uses the PhysicAI stats. Never share a scaler.
    """
    if windows.size == 0:
        raise ValueError("Cannot compute normalization statistics for empty windows.")
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std = windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def static_window_mask(windows: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Mark windows whose every channel barely moves (peak-to-peak < threshold)."""
    if windows.size == 0:
        return np.zeros((0,), dtype=bool)
    movement = np.max(np.ptp(windows, axis=1), axis=1)
    return movement < threshold
