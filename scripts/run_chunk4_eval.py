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
from src.models.bc_chunk import BCChunkMLP  # noqa: E402
from src.models.flow_matching import (  # noqa: E402
    ConditionalFlowConfig,
    ConditionalFlowField,
    conditional_flow_loss,
    conditional_sample,
)
from src.train.train_lstm_vae import episode_split  # noqa: E402
from run_conditional_flow import build_pairs, jerk, load_physicai_episodes, norm_stats  # noqa: E402


def train_fm(loader, cfg, k, epochs, lam, device, seed):
    torch.manual_seed(seed)
    fm = ConditionalFlowField(ConditionalFlowConfig(
        action_dim=6, state_dim=6, chunk_size=k, hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"], kernel_size=cfg["kernel_size"], embed_dim=cfg["embed_dim"],
    )).to(device)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    for _ in range(epochs):
        fm.train()
        for bs, bc in loader:
            bs, bc = bs.to(device), bc.to(device)
            loss = conditional_flow_loss(fm, bc, bs, lambda_smooth=lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 5.0); opt.step()
    fm.eval()
    return fm


def train_bc(loader, cfg, k, epochs, device, seed):
    torch.manual_seed(seed + 7)
    bc = BCChunkMLP(in_dim=6, action_dim=6, chunk_size=k, hidden_dim=cfg["hidden_dim"]).to(device)
    opt = torch.optim.Adam(bc.parameters(), lr=cfg["lr"]); lossf = torch.nn.MSELoss()
    for _ in range(epochs):
        bc.train()
        for bs, bck in loader:
            bs, bck = bs.to(device), bck.to(device)
            opt.zero_grad(); lossf(bc(bs), bck.reshape(bck.size(0), -1)).backward(); opt.step()
    bc.eval()
    return bc


def ci95(x):
    x = np.asarray(x)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main() -> None:
    p = argparse.ArgumentParser(description="Chunk 4: multi-seed robustness + distribution-level generation metrics.")
    p.add_argument("--config", default="configs/conditional_flow_physicai.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pr-k", type=int, default=3, help="k for precision/recall k-NN radius.")
    p.add_argument("--dist-samples", type=int, default=1000, help="pool size for distribution metrics.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    k = cfg["chunk_size"]; lam = cfg.get("lambda_smooth", 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = load_physicai_episodes(cfg["physicai_root"])
    win_ep = np.repeat(np.arange(len(episodes)), [len(e) for e in episodes])
    print(f"physicai {len(episodes)} ep | k={k} | lambda_smooth={lam} | epochs={epochs} | seeds={args.seeds} | {device}\n")

    rows = {m: {"FM": [], "BC": []} for m in
            ["jerk_ratio", "inside", "anchor", "chunkMSE", "wass", "precision", "recall"]}
    rows["diversity"] = {"FM": []}
    rows["best_of_n"] = {"FM": []}; rows["bc_mse_d"] = {"BC": []}

    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        tr_idx, _, _, _ = episode_split(win_ep, cfg["val_frac"], seed)
        train_eps = sorted(set(win_ep[tr_idx].tolist()))
        test_eps = [e for e in range(len(episodes)) if e not in set(train_eps)]
        s_tr, c_tr = build_pairs(episodes, train_eps, k)
        s_te, c_te = build_pairs(episodes, test_eps, k)
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
            fm_gen = dA(conditional_sample(fm, cond, num_steps=cfg["sample_steps"]).cpu().numpy())
            bc_gen = dA(bc(cond).reshape(-1, k, 6).cpu().numpy())

        for name, gen, model in [("FM", fm_gen, "FM"), ("BC", bc_gen, "BC")]:
            rows["jerk_ratio"][model].append(jerk(gen) / rj)
            rows["inside"][model].append(float(((gen >= amin) & (gen <= amax)).mean()))
            rows["anchor"][model].append(float(np.abs(gen[:, 0, :] - s_ev).mean()))
            rows["chunkMSE"][model].append(float(((gen - c_ev) ** 2).mean()))

        # distribution-level: pooled generated vs real (held-out) chunks
        m = min(args.dist_samples, len(idx))
        real_pool = c_ev[:m]
        for model, gen in [("FM", fm_gen[:m]), ("BC", bc_gen[:m])]:
            rows["wass"][model].append(per_joint_wasserstein(gen, real_pool))
            prec, rec = precision_recall(real_pool.reshape(m, -1), gen.reshape(m, -1), k=args.pr_k)
            rows["precision"][model].append(prec); rows["recall"][model].append(rec)

        # multimodality (FM): N samples per condition on a subset
        n_div = min(cfg["diversity_conditions"], len(idx)); N = cfg["diversity_samples"]
        cond_d = cond[:n_div]
        with torch.no_grad():
            stacks = np.stack([dA(conditional_sample(fm, cond_d, num_steps=cfg["sample_steps"]).cpu().numpy())
                               for _ in range(N)], 0)
        rows["diversity"]["FM"].append(float(stacks.std(0).mean()))
        rows["best_of_n"]["FM"].append(float(((stacks - c_ev[:n_div][None]) ** 2).mean(axis=(2, 3)).min(0).mean()))
        rows["bc_mse_d"]["BC"].append(float(((bc_gen[:n_div] - c_ev[:n_div]) ** 2).mean()))
        print(f"seed {seed}: FM jerk={rows['jerk_ratio']['FM'][-1]:.2f}x wass={rows['wass']['FM'][-1]:.3f} "
              f"prec={rows['precision']['FM'][-1]:.2f} rec={rows['recall']['FM'][-1]:.2f} | "
              f"BC wass={rows['wass']['BC'][-1]:.3f} prec={rows['precision']['BC'][-1]:.2f} rec={rows['recall']['BC'][-1]:.2f}")

    def fmt(key, model):
        v = np.array(rows[key][model])
        return f"{v.mean():.3f}+/-{ci95(v):.3f}"

    print(f"\n===== CHUNK 4: {len(args.seeds)}-seed mean+/-95%CI (held-out, raw units) =====")
    print(f"{'metric':<12} {'FM':>16} {'BC':>16}")
    for key in ["jerk_ratio", "inside", "anchor", "chunkMSE", "wass", "precision", "recall"]:
        print(f"{key:<12} {fmt(key,'FM'):>16} {fmt(key,'BC'):>16}")
    print(f"{'diversity':<12} {fmt('diversity','FM'):>16} {'(BC=0)':>16}")
    print(f"{'best_of_N':<12} {fmt('best_of_n','FM'):>16} {fmt('bc_mse_d','BC'):>16}")
    print("\nReading: lower=better for jerk_ratio/anchor/chunkMSE/wass/best_of_N; higher=better for "
          "inside/precision/recall/diversity. precision=realism, recall=coverage.")


if __name__ == "__main__":
    main()
