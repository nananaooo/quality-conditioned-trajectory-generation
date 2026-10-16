# quality-conditioned-trajectory-generation

Code for the paper

> M. Kim, Y. Choi, D. Ryu, J. Baik, and S. Kim, "Quality Conditioning for Robot Data Generation from Imperfect
> Demonstrations," submitted to *Journal of Web Engineering*, 2026.

The paper trains a flow-matching generator of robot action chunks (k = 8 joint targets) on **all** teleoperation
demonstrations, conditioned on two label-free quality axes computed from each chunk (activity and smoothness).
At generation time, the quality condition is drawn from usable chunks near the query state. The generator is compared
with filter-then-train baselines on four public LeRobot corpora.

This repository contains the model and metric code, the scripts that produced every table and figure, pinned data
download scripts, and the per-seed results behind the paper's numbers.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # install torch for your CUDA version from pytorch.org
python scripts/prepare_data.py                         # pinned download (joint data only) + windows
```

The paper used Python 3.13.0, PyTorch 2.6.0 (CUDA 12.4), one NVIDIA RTX 3070 Ti, Windows 11.

### Data

The datasets are **not redistributed** here. `scripts/prepare_data.py` downloads only `data/` and `meta/` (no video)
at the exact commits used in the paper and builds windows of length 60 with stride 30 that never cross episodes.
Please check each dataset's terms on the Hugging Face Hub.

| Corpus (code name) | Hub repository | Commit |
|---|---|---|
| being_so101 | `BeingBeyond/Being-H-EDU_SO101` (`pick_cube_plate_v3.0`) | `8bd76e3` |
| rayenghali | `Rayenghali/combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30` | `c1250a1` |
| koch (`koch_pickplace5`) | `lerobot/koch_pick_place_5_lego` | `af1f4ff` |
| aloha (`aloha_tape`) | `lerobot/aloha_static_tape` | `9295587` |

Re-running `prepare_data.py` reproduces the window arrays used in the paper bit for bit.

The authors' PhysicAI-Arm data, used only in two preliminary validity checks of the appendix, is not included.

## Reproducing the paper

`run_all.sh` runs every step in order. The steps are resumable, and the whole run takes about 13 h on one
RTX 3070 Ti. Results are written to `outputs/`.

```bash
bash run_all.sh
```

| Paper item | Produced by |
|---|---|
| Table 1 (chunks, unique chunks) | `review_08_dedup.py` |
| Table 2, Appendix Table A2(a) | `review_14_k64.py` → `review_14_aggregate.py` (section h), exact values `review_15b.py` |
| Figure 2 (temperature trade-off) | `review_dedup_reeval.py` (F curve), `review_14_k64.py` (QC curve) → `make_figures.py` |
| Table 3, Appendix Table A2(b) | `review_15b.py` (b), `review_14_aggregate.py` (a–c) |
| Table 4, Appendix Tables A3, A4 | `review_12_lowq_axes.py`, `review_14_k64.py` → `review_14_aggregate.py` (e), `review_15b.py` (d) |
| Table 5, Figure 3, Appendix Table A6 | `review_11_rq.py`, `review_14_k64.py --stage scales` → `review_15b.py` (a), `make_figures.py` |
| Appendix Table A5(a) (request schemes) | `review_dedup_reeval.py` → `review_15b.py` (c) |
| Appendix Table A5(b) (choice of K) | `review_13_request.py`, `review_13_select.py`, `review_13_aggregate.py` |
| Appendix Table A7 (95% CIs) | `review_20_ci.py` |
| Appendix Table A1 (controllability) | `run_review_campaign.py` → `review_campaign_aggregate.py` |
| Appendix defect-injection check | `review_A5_defect_matrix.py` |
| Section 4.5 generation speed | `review_20_timing.py` |

Configuration: `configs/conditional_flow_physicai.yaml` (model, training, sampling) and `configs/review_campaign.yaml`
(corpora, seeds 0–4, data fractions, evaluation settings). The model has about 450,000 parameters and is trained for
100 epochs with Adam (5e-4), batch 256, and gradient clipping at 5. The last epoch is evaluated; there is no early
stopping or checkpoint selection.

## Names in the code

The scripts were written over several analysis rounds, and their docstrings refer to these rounds as "report NN".
Configuration names in the code map to the paper as follows:

| Code | Paper |
|---|---|
| `A` | F (filtering: activity top 50 %) |
| `A_step` | F, compute-matched (same number of updates as QC) |
| `A_Tmatch*` | F, temperature-matched (interpolated on the F temperature curve) |
| `A25` / `A75` / `A_con` | F keeping the activity top 75 % / top 25 % / action-consistency top 50 % |
| `C` | FU (unconditioned, all data) |
| `OutFilt` | OF (FU + rejection of chunks below τ) |
| `B1` + state-near request (`ACs`, `AC64`) | AC (activity-only condition) |
| `B2` | the quality-conditioned model |
| `B2` + `QC64` request (K = 64) | **QC** (final method) |
| `B2` + distribution draw (`B2`, "QC-M") | state-agnostic request |
| `mode:global_median` / `mode:per_state_fixed` | global fixed / per-state fixed request |
| `QCs` | state-near request with K = 32 (before K was selected) |
| `R`, `R2` | reference set of usable held-out chunks / two-axis reference set |

## Results included

- `results/paper_tables_per_seed.csv`: per-seed values behind Tables 2–5 and Appendix Tables A4–A6.
- `results/k64_per_seed.csv`: the final QC (K = 64) and AC configurations.
- `results/rq_per_seed_k32_and_baselines.csv`: all baselines (F and variants, FU, OF, request schemes) and the earlier
  K = 32 request.
- `results/defects_r2_consistency_per_seed.csv`: defect rates, the R2 set, and the consistency target.
- `results/k_selection_per_seed.csv` and `results/k_selection_preregistration.md`: the validation-split choice of K.
- `results/tableA7_ci.csv`: mean paired differences with 95% CIs.
- `results/generation_timing.csv`: generation latency and throughput.
- `figures/`: Figures 2 and 3 (PDF/PNG) and their plotted values.

## Citation

```bibtex
@unpublished{kim2026qualityconditioning,
  title  = {Quality Conditioning for Robot Data Generation from Imperfect Demonstrations},
  author = {Kim, Minjae and Choi, Yunseo and Ryu, Duksan and Baik, Jongmoon and Kim, Suntae},
  note   = {Submitted to the Journal of Web Engineering},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The datasets are subject to their own terms on the
Hugging Face Hub.
