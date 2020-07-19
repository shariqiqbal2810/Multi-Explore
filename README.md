# Multi-Explore
Code for [*Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1905.12127) (Iqbal and Sha, arXiv 1905.12127)

## Requirements
* Python 3.7.3
* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* [PyTorch](http://pytorch.org/), version: 1.2.0
* [OpenAI Gym](https://github.com/openai/gym), version: 0.14.0
* [ViZDoom](https://github.com/mwydmuch/ViZDoom), version: 1.1.7
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 1.14.0 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.8 (for logging)

The versions are what were used in this project but are not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

All hyperparameters can be found in the Appendix of the paper. Default hyperparameters are for Task 1 in the GridWorld environment using 2 agents.

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```
@article{iqbal2019coordinated,
  title={Coordinated Exploration via Intrinsic Rewards for Multi-Agent Reinforcement Learning},
  author={Iqbal, Shariq and Sha, Fei},
  journal={arXiv preprint arXiv:1905.12127},
  year={2019}
}
```
