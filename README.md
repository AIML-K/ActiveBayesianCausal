# ABC3: Active Bayesian Causal Inference with Cohn Criteria in Randomized Experiments (AAAI 2025)

This is a code for ABC3: Active Bayesian Causal Inference with Cohn Criteria in Randomized Experiments.

[Paper](https://arxiv.org/abs/2412.11104)

## Abstract

In causal inference, a randomized experiment is a de facto method to overcome various theoretical issues in observational study. However, the experimental design requires expensive costs, so an efficient experimental design is necessary. We propose ABC3, a Bayesian active learning policy for causal inference. We show a policy minimizing an estimation error on conditional average treatment effect is equivalent to minimizing an integrated posterior variance, similar to Cohn criteria. We theoretically prove ABC3 also minimizes an imbalance between the treatment and control groups and the type 1 error probability. Imbalance-minimizing characteristic is especially notable as several works have emphasized the importance of achieving balance. Through extensive experiments on real-world data sets, ABC3 achieves the highest efficiency, while empirically showing the theoretical results hold.

## Installation

1. Install the requirements by `pip install -r requirements.txt`

2. Install `torch` following the [instruction](https://pytorch.org/get-started/locally/)

## Main Experiments

1. Run the `run.sh` by `bash run.sh`

2. Check the resulting plots in `plots/cate_error`, `plots/mmd` and `plots/type1`

## Other Experiments

- If you are interested in reproducing the hyperparameter test (Fig. 4 and 5), run `run/hyperparameter_kernel.py` and `hyperparameter_sigma.py` with desired arguments, then run `src/plot.py` with arguments `kernel` or `sigma`.

- If you want to check our assumption (Fig. 6), run `run/assumption.py`, then run `src/plot.py` wirh a `assumption` argument.

- If you want to test our sampling-and-optimization-based algorithm (Appendix E), run `run/sampling.py` with proper arguments, then run `src/plot.py` with a `sampling` argument.

- If you want to plug-in different regressors (Appendix F), run `run/regressor.py` with proper arguments, then run `src/plot.py` with a `reg` arguement.
