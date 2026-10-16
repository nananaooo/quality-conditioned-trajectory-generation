from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_project_root

add_project_root()

from src.data.lerobot_reader import (  # noqa: E402
    find_dataset_roots,
    group_episode_frames,
    load_info,
    read_data_frame,
    sort_frames,
    stack_feature_matrix,
)
from src.data.windowing import (  # noqa: E402
    normalization_stats,
    slice_episode,
    static_window_mask,
)


def expand_feature_names(features: list[str], info: dict) -> list[str]:
    """Build per-channel labels like 'observation.state/shoulder_pan.pos'."""
    feat_info = info.get("features", {}) if isinstance(info, dict) else {}
    names: list[str] = []
    for feature in features:
        joint_names = feat_info.get(feature, {}).get("names")
        if isinstance(joint_names, list):
            names.extend(f"{feature}/{j}" for j in joint_names)
        else:
            names.append(feature)
    return names


def apply_quality_filter(windows: np.ndarray, mode: str) -> np.ndarray:
    """Optional quality-aware filter hook (reserved for RQ3).

    Currently only 'none' is supported. The quality-aware filtering will be
    designed later to fit this research; do NOT hardcode prior-paper rules here.
    """
    if mode == "none":
        return np.ones((windows.shape[0],), dtype=bool)
    raise NotImplementedError(
        f"Quality filter '{mode}' is not implemented yet (reserved hook for RQ3)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sliding windows of (observation.state, action) from one or "
        "more LeRobot datasets. Windows never cross episode boundaries."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="A LeRobot dataset root, or a parent dir of several single-dataset roots "
        "(e.g. data_raw/physicai with 20 session subdirs).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["observation.state", "action"],
        help="Vector features to concatenate per frame (default: state+action = 12D).",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--static-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--quality-filter",
        default="none",
        help="Reserved hook for RQ3 quality-aware filtering. Only 'none' for now.",
    )
    args = parser.parse_args()

    roots = find_dataset_roots(args.root)
    if not roots:
        raise FileNotFoundError(f"No LeRobot dataset roots found under: {args.root}")

    all_windows: list[np.ndarray] = []
    window_episode: list[int] = []  # global episode id per window (for leakage-free splits)
    global_ep = 0
    for root in roots:
        df = read_data_frame(root)
        missing = [f for f in args.features if f not in df.columns]
        if missing:
            raise KeyError(f"{root}: missing required columns: {missing}")
        for _, ep_df in group_episode_frames(sort_frames(df)):
            values = stack_feature_matrix(ep_df, args.features)
            wins = slice_episode(values, args.window_size, args.stride)
            all_windows.extend(wins)
            window_episode.extend([global_ep] * len(wins))
            global_ep += 1

    if not all_windows:
        raise ValueError("No windows produced. Check episode lengths, window size, stride.")

    n_episodes = global_ep
    windows = np.asarray(all_windows, dtype=np.float32)
    window_episode_arr = np.asarray(window_episode, dtype=np.int64)
    if not np.isfinite(windows).all():
        bad = int((~np.isfinite(windows)).sum())
        raise ValueError(f"Windows contain NaN or inf values: {bad}")

    keep_mask = apply_quality_filter(windows, args.quality_filter)
    windows = windows[keep_mask]
    window_episode_arr = window_episode_arr[keep_mask]

    mean, std = normalization_stats(windows)
    static_mask = static_window_mask(windows, threshold=args.static_threshold)
    feature_names = expand_feature_names(args.features, load_info(roots[0]))

    output = Path(args.output)
    stats_output = Path(args.stats_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, windows)
    np.savez(
        stats_output,
        mean=mean,
        std=std,
        static_mask=static_mask,
        static_ratio=np.asarray(static_mask.mean(), dtype=np.float32),
        window_size=np.asarray(args.window_size),
        stride=np.asarray(args.stride),
        features=np.asarray(args.features),
        feature_names=np.asarray(feature_names),
        window_episode=window_episode_arr,
        n_episodes=np.asarray(n_episodes),
    )

    print(f"Datasets found:   {len(roots)} root(s), {n_episodes} episode(s)")
    print(f"Features:         {args.features} -> {windows.shape[2]}D")
    print(f"Saved windows:    {output}  shape={list(windows.shape)}")
    print(f"Saved stats:      {stats_output}  (mean/std shape={list(mean.shape)})")
    print(f"NaN/inf:          none")
    print(
        f"Static windows:   {int(static_mask.sum())}/{len(static_mask)} "
        f"({static_mask.mean() * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
