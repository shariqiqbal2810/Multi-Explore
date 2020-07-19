from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from itertools import chain
from utils.misc import hard_update
from utils.policies import DiscretePolicy
import torch.nn.functional as F

class Agent(object):
    """
    General class for agents (policy, target policy, etc)
    """
    def __init__(self, obs_shape, action_size, hidden_dim=64,
                 lr=0.01, adam_eps=1e-8, nonlin=F.relu, n_pol_heads=1):
        self.policy = DiscretePolicy(obs_shape,
                                     action_size,
                                     hidden_dim=hidden_dim,
                                     nonlin=nonlin,
                                     n_heads=n_pol_heads)
        self.target_policy = DiscretePolicy(obs_shape,
                                            action_size,
                                            hidden_dim=hidden_dim,
                                            nonlin=nonlin,
                                            n_heads=n_pol_heads)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr, eps=adam_eps)

    def step(self, obs, explore=False, head=0):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
            head (int): Which policy head to use
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, sample=explore, head=head)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params, load_ir=False):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
