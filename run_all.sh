#!/usr/bin/env bash
# Full reproduction of the paper's experiments, in order. Every step is resumable (finished cells are skipped).
# Approximate time on one RTX 3070 Ti (from the original runs): ~13 h in total, most of it in steps 2, 4 and 5.
set -euo pipefail
PY=${PY:-python}

# 1. data: pinned downloads (joint data only) + windows (length 60, stride 30)                       ~ minutes
$PY scripts/prepare_data.py

# 2. training campaign: 4 corpora x 5 seeds x {1, 1/4} data; arms F, update-matched F, FU, AC, QC (+ z-axis pair)
#    also writes the controllability sweep of Appendix Table A1                                        ~7.7 h
$PY scripts/run_review_campaign.py --config configs/review_campaign.yaml

# 3. evaluation on the de-duplicated target R (T = 1 metrics, OF, request-scheme ablation, temperature curves)  ~0.5 h
$PY scripts/review_dedup_reeval.py

# 4. filtering variants: activity top 75 % / top 25 %, consistency top 50 % (trained here)            ~2 h
$PY scripts/review_dedup_filter_variants.py

# 5. common target R for all data fractions; trains F and QC at 1/2 and 1/8 (nested episode subsets)  ~1.8 h
$PY scripts/review_11_rq.py --stage full
$PY scripts/review_11_rq.py --stage common --scales 0.25
$PY scripts/review_11_rq.py --stage common --scales 0.5 0.125

# 6. defect rates, two-axis reference set R2, consistency target                                       ~0.3 h
$PY scripts/review_12_lowq_axes.py

# 7. choice of the number of neighbours K on a validation half of the training episodes (pre-registered rule)
$PY scripts/review_13_request.py --stage val
$PY scripts/review_13_select.py                                   # -> selected K = 64 (uniform)
$PY scripts/review_13_request.py --stage confirm --k 64 --weighted 0

# 8. final QC (K = 64; K proportional to the training pool at 1/2, 1/4, 1/8: 32, 16, 8)              ~0.3 h
$PY scripts/review_14_k64.py --stage full
$PY scripts/review_14_k64.py --stage scales

# 9. aggregation -> tables
$PY scripts/review_campaign_aggregate.py                          # Appendix Table A1 (controllability)
$PY scripts/review_dedup_aggregate.py
TRADEOFF_DIR=outputs/campaign_dedup/tradeoff $PY scripts/review_tradeoff_aggregate.py
$PY scripts/review_11_aggregate.py
$PY scripts/review_12_aggregate.py
$PY scripts/review_13_aggregate.py                                # Appendix Table A5(b)
$PY scripts/review_14_aggregate.py                                # Tables 2-5, A2, A3 (run-level tests)
$PY scripts/review_15b.py                                         # exact values for Tables 2-5, A4, A5(a), A6
$PY scripts/review_20_ci.py                                       # Appendix Table A7
$PY scripts/review_08_dedup.py                                    # Table 1 (chunks / unique chunks)

# 10. figures and generation timing
$PY scripts/make_figures.py                                       # Figures 2 and 3 -> figures/
$PY scripts/review_20_timing.py                                   # Section 4.5 generation speed

# Preliminary checks of the appendix (axis validity). The defect-injection test uses the public corpora:
$PY scripts/review_A5_defect_matrix.py
# The PhysicAI-Arm checks (run_chunk16_quality_validity.py, review_12b_workshop_rule.py) need the authors' PhysicAI-Arm
# data, which is not part of this repository.
