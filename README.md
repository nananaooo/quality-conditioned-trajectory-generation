# quality-conditioned-trajectory-generation
Code for training a quality-conditioned flow-matching generator of robot
joint-trajectory chunks on full teleoperation data, and comparing it with
filter-then-generate baselines.
**Status:** code release in preparation (expected by <YYYY-MM-DD>).
Planned contents:
- model and metric code (flow-matching generator, precision/recall, diversity, best-of-N)
- training and evaluation scripts with fixed seeds and configs
- download scripts for the four public LeRobot datasets (pinned revisions; data are not redistributed)
- per-seed result CSVs and figure scripts
