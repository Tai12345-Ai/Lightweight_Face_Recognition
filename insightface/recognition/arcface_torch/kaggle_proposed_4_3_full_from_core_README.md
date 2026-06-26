# Proposed 4.3 Full from Core

This file documents the intended Kaggle workflow:

1. Train Proposed 4.3 Core and save `best.pth` or `latest.pt`.
2. Add the Core checkpoint to Kaggle input.
3. Run `kaggle_proposed_4_3_full_5eval_degraded_runner.py`.
4. The runner should warm-start Full from Core, build or reuse multi-UI centers, train Full, and evaluate clean plus degraded severities `1,3,5`.
