"""Chunk 6: de-risk direction A. Does the conditional-FM advantage over chunk-BC actually
OPEN UP on a genuinely multimodal dataset (being_so101) vs the parity seen on near-deterministic
PhysicAI (chunk 4)?  Same FM(lambda=5) vs BC protocol + distribution metrics, on window data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import add_project_root

add_project_root()

from src.eval.distribution_metrics import per_joint_wasserstein, precision_recall  # noqa: E402
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import jerk, norm_stats  # noqa: E402
from run_chunk4_eval import ci95, train_bc, train_fm  # noqa: E402


def build_pairs_from_windows(windows, k, state_dim=6):
    """(state[t], action[t:t+k]) within each window (windows already respect episode boundaries).

    state_dim: leading channels = state, remainder = action (6 for SO101/Koch, 14 for ALOHA)."""
    states, chunks = [], []
    for w in windows:
        for t in range(w.shape[0] - k):
            states.append(w[t, :state_dim]); chunks.append(w[t:t + k, state_dim:])
    return np.asarray(states, np.float32), np.asarray(chunks, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/conditional_flow_physicai.yaml", help="hyperparams (data paths ignored)")
    p.add_argument("--windows", default="data_processed/windows/being_so101_stateaction_w60_s30.npy")
    p.add_argument("--stats", default="data_processed/stats/being_so101_stateaction_w60_s30_stats.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pr-k", type=int, default=3)
    p.add_argument("--dist-samples", type=int, default=1000)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    k = cfg["chunk_size"]; lam = cfg.get("lambda_smooth", 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = np.load(args.windows).astype(np.float32)
    stats = np.load(args.stats, allow_pickle=True)
    win_ep = stats["window_episode"]
    n_eps = int(stats["n_episodes"]) if "n_episodes" in stats else len(np.unique(win_ep))
    print(f"{Path(args.windows).stem} | {windows.shape} | {n_eps} eps | k={k} lambda={lam} epochs={epochs} "
          f"seeds={args.seeds} | {device}\n")

    rows = {m: {"FM": [], "BC": []} for m in
            ["jerk_ratio", "inside", "chunkMSE", "wass", "precision", "recall"]}
    rows["diversity"] = {"FM": []}; rows["best_of_n"] = {"FM": []}; rows["bc_mse_d"] = {"BC": []}

    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        tr_w, te_w, _, _ = episode_split(win_ep, cfg["val_frac"], seed)
        s_tr, c_tr = build_pairs_from_windows(windows[tr_w], k)
        s_te, c_te = build_pairs_from_windows(windows[te_w], k)
        s_mean, s_std = norm_stats(s_tr); a_mean, a_std = norm_stats(c_tr)
        nS = lambda x: (x - s_mean) / s_std
        dA = lambda x: x * a_std + a_mean
        loader = DataLoader(TensorDataset(torch.from_numpy(nS(s_tr)), torch.from_numpy((c_tr - a_mean) / a_std)),
                            batch_size=cfg["batch_size"], shuffle=True)

        fm = train_fm(loader, cfg, k, epochs, lam, device, seed)
        bc = train_bc(loader, cfg, k, epochs, device, seed)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(s_te))[: cfg["eval_conditions"]]
        s_ev, c_ev = s_te[idx], c_te[idx]
        cond = torch.from_numpy(nS(s_ev)).to(device)
        rj = jerk(c_ev); amin, amax = c_tr.reshape(-1, 6).min(0), c_tr.reshape(-1, 6).max(0)
        with torch.no_grad():
            fm_gen = dA(conditional_sample_fm(fm, cond, cfg))
            bc_gen = dA(bc(cond).reshape(-1, k, 6).cpu().numpy())
        for model, gen in [("FM", fm_gen), ("BC", bc_gen)]:
            rows["jerk_ratio"][model].append(jerk(gen) / rj)
            rows["inside"][model].append(float(((gen >= amin) & (gen <= amax)).mean()))
            rows["chunkMSE"][model].append(float(((gen - c_ev) ** 2).mean()))
        m = min(args.dist_samples, len(idx)); real_pool = c_ev[:m]
        for model, gen in [("FM", fm_gen[:m]), ("BC", bc_gen[:m])]:
            rows["wass"][model].append(per_joint_wasserstein(gen, real_pool))
            pr, rc = precision_recall(real_pool.reshape(m, -1), gen.reshape(m, -1), k=args.pr_k)
            rows["precision"][model].append(pr); rows["recall"][model].append(rc)

        n_div = min(cfg["diversity_conditions"], len(idx)); N = cfg["diversity_samples"]
        cond_d = cond[:n_div]
        with torch.no_grad():
            stacks = np.stack([dA(conditional_sample_fm(fm, cond_d, cfg)) for _ in range(N)], 0)
        rows["diversity"]["FM"].append(float(stacks.std(0).mean()))
        rows["best_of_n"]["FM"].append(float(((stacks - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean()))
        rows["bc_mse_d"]["BC"].append(float(((bc_gen[:n_div] - c_ev[:n_div]) ** 2).mean()))
        print(f"seed {seed}: FM chunkMSE={rows['chunkMSE']['FM'][-1]:.2f} recall={rows['recall']['FM'][-1]:.2f} "
              f"prec={rows['precision']['FM'][-1]:.2f} jerk={rows['jerk_ratio']['FM'][-1]:.2f}x | "
              f"BC chunkMSE={rows['chunkMSE']['BC'][-1]:.2f} recall={rows['recall']['BC'][-1]:.2f}")

    def fmt(key, model):
        v = np.array(rows[key][model]); return f"{v.mean():.3f}+/-{ci95(v):.3f}"

    print(f"\n===== CHUNK 6 ({Path(args.windows).stem}): {len(args.seeds)}-seed FM(lambda={lam}) vs chunk-BC =====")
    print(f"{'metric':<12}{'FM':>16}{'BC':>16}")
    for key in ["jerk_ratio", "inside", "chunkMSE", "wass", "precision", "recall"]:
        print(f"{key:<12}{fmt(key,'FM'):>16}{fmt(key,'BC'):>16}")
    print(f"{'diversity':<12}{fmt('diversity','FM'):>16}{'(BC=0)':>16}")
    print(f"{'best_of_N':<12}{fmt('best_of_n','FM'):>16}{fmt('bc_mse_d','BC'):>16}")
    print("\nGO(A): FM advantage over BC clearly LARGER here than on PhysicAI (ch4) -- esp chunkMSE gap opens, "
          "recall gap widens. lower=better jerk/chunkMSE/wass/best_of_N; higher=better inside/precision/recall.")


def conditional_sample_fm(fm, cond, cfg):
    from src.models.flow_matching import conditional_sample
    return conditional_sample(fm, cond, num_steps=cfg["sample_steps"]).cpu().numpy()


if __name__ == "__main__":
    main()
