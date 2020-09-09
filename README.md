# Multi-Explore
Code for [*Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1905.12127) (Iqbal and Sha, arXiv 1905.12127)

## Requirements
Conda environment specification is located in `environment.yml`.
Use this file to manually install dependencies if desired.
Otherwise, follow instructions in the next section.

## How to Run
Install conda environment with all dependencies
```shell
conda env create -f environment.yml
```

Activate environment
```shell
conda activate multi-explore
```

All training code is contained within `main.py`. To view options simply run:

```shell
python main.py --help
```

All hyperparameters can be found in the Appendix of the paper. Default hyperparameters are for Task 1 in the GridWorld environment using 2 agents.
For Flip-Task include the flags `--task_config 4 --map_ind -1`.

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@article{iqbal2019coordinated,
  title={Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning},
  author={Iqbal, Shariq and Sha, Fei},
  journal={arXiv preprint arXiv:1905.12127},
  year={2019}
}
```
