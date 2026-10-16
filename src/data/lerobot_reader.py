from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("observation.state", "action")


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    shape: list[int] | None
    names: list[str] | None


def load_info(root: str | Path) -> dict[str, Any]:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_parquet_files(root: str | Path, subdir: str = "data") -> list[Path]:
    base = Path(root) / subdir
    if not base.exists():
        return []
    return sorted(base.rglob("*.parquet"))


def read_parquet_files(files: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except Exception:
            if columns is None:
                raise
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_data_frame(root: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return read_parquet_files(find_parquet_files(root, "data"), columns=columns)


def read_tasks(root: str | Path) -> pd.DataFrame:
    path = Path(root) / "meta" / "tasks.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_episodes(root: str | Path) -> pd.DataFrame:
    files = find_parquet_files(Path(root) / "meta", "episodes")
    return read_parquet_files(files)


def feature_info(info: dict[str, Any], feature_name: str) -> FeatureInfo:
    features = info.get("features", {})
    raw = features.get(feature_name, {}) if isinstance(features, dict) else {}
    shape = raw.get("shape")
    names = raw.get("names")
    if isinstance(names, dict):
        names = names.get("motors") or names.get("joints") or list(names.values())
    return FeatureInfo(
        name=feature_name,
        shape=list(shape) if isinstance(shape, list) else None,
        names=list(names) if isinstance(names, list) else None,
    )


def values_to_matrix(series: pd.Series) -> np.ndarray:
    values = series.to_list()
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray([np.asarray(v, dtype=np.float32) for v in values], dtype=np.float32)


def stack_feature_matrix(df: pd.DataFrame, features: list[str] | tuple[str, ...]) -> np.ndarray:
    """Concatenate several vector features column-wise into a [T, sum(dims)] matrix.

    Used to build the (observation.state, action) = 12D generation target.
    """
    matrices = [values_to_matrix(df[feature]) for feature in features]
    return np.concatenate(matrices, axis=1).astype(np.float32)


def find_dataset_roots(root: str | Path) -> list[Path]:
    """Return LeRobot dataset roots under ``root``.

    A LeRobot dataset root is a directory that directly contains a ``data/`` folder.
    If ``root`` itself is such a root, returns ``[root]``. Otherwise treats ``root``
    as a parent of several single-dataset subdirectories (e.g. PhysicAI's 20 sessions).
    """
    root = Path(root)
    if (root / "data").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "data").is_dir())


def sort_frames(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ("episode_index", "frame_index", "timestamp", "index") if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def group_episode_frames(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    df = sort_frames(df)
    if "episode_index" not in df.columns:
        return [(0, df)]
    return [(int(idx), group.reset_index(drop=True)) for idx, group in df.groupby("episode_index", sort=True)]


def summarize_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    if matrix.size == 0:
        return {}
    return {
        "min": np.nanmin(matrix, axis=0),
        "max": np.nanmax(matrix, axis=0),
        "mean": np.nanmean(matrix, axis=0),
        "std": np.nanstd(matrix, axis=0),
    }


def episode_lengths(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=np.int64)
    if "episode_index" not in df.columns:
        return pd.Series([len(df)], index=[0])
    return df.groupby("episode_index").size()
