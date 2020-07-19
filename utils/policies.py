import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import onehot_from_logits, categorical_sample
from .networks import CombineNet, MLPNet


class HeadSelector(nn.Module):
    """
    Contains parameter vector that learns head selector policy
    """
    def __init__(self, nagents, n_pol_heads):
        super(HeadSelector, self).__init__()
        self.nagents = nagents
        self.n_pol_heads = n_pol_heads

        self.selector = nn.Parameter(torch.randn(1, n_pol_heads) * 0.01)

    def forward(self, sample=True, return_all_probs=False,
                return_all_log_probs=False,
                return_log_pi=False, return_entropy=False):
        probs = F.softmax(self.selector, dim=1)
        if sample:
            heads, _ = categorical_sample(probs)
        else:
            heads = probs.max(dim=1, keepdim=True)[1]
        rets = [heads]
        if return_log_pi or return_all_log_probs or return_entropy:
            log_probs = F.log_softmax(self.selector, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_all_log_probs:
            rets.append(log_probs)
        if return_log_pi:
            # return log probability of selected head
            rets.append(log_probs.gather(1, heads))
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1))
        if len(rets) == 1:
            return rets[0]
        return rets


class DiscretePolicy(nn.Module):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, obs_shape, *args, **kwargs):
        super(DiscretePolicy, self).__init__()
        if type(obs_shape[0]) is int:
            if len(obs_shape) == 1:
                self.network = MLPNet(obs_shape[0], *args, **kwargs)
            else:
                raise NotImplementedError('Purely image observations not implemented')
        else:
            self.network = CombineNet(obs_shape, *args, **kwargs)

    def scale_shared_grads(self):
        self.network.scale_shared_grads()

    def process_logits(self, logits, sample=True, return_all_probs=False,
                       return_all_log_probs=False,
                       return_log_pi=False, regularize=False,
                       return_entropy=False):
        """
        Return desired values for output of single policy head
        """
        probs = F.softmax(logits, dim=1)
        if sample:
            int_act, act = categorical_sample(probs)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_all_log_probs or return_entropy:
            log_probs = F.log_softmax(logits, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_all_log_probs:
            rets.append(log_probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(logits**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets

    def forward(self, obs, head=None, **kwargs):
        out = self.network.forward(obs, head=head)
        if isinstance(out, list):
            return [self.process_logits(l, **kwargs) for l in out]
        return self.process_logits(out, **kwargs)
