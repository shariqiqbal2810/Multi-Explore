import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class CentralCritic(nn.Module):
    def __init__(self, vect_state_size, action_size,
                 nagents, hidden_dim=64, nonlin=F.relu, n_intr_rew_heads=0,
                 sep_extr_head=True):
        super(CentralCritic, self).__init__()
        assert (sep_extr_head or n_intr_rew_heads > 0)
        self.vect_state_size = vect_state_size
        self.action_size = action_size
        self.nagents = nagents
        self.nonlin = nonlin
        self.n_intr_rew_heads = n_intr_rew_heads
        self.sep_extr_head = sep_extr_head
        self.n_pol_heads = self.n_intr_rew_heads + int(self.sep_extr_head)

        self.state_vect_encoder = nn.Sequential()
        self.state_vect_encoder.add_module('vect_enc_fc',
                                           nn.Linear(vect_state_size,
                                                     hidden_dim))
        self.state_vect_encoder.add_module('vect_enc_nl', nn.ReLU())

        self.fully_shared_modules = [self.state_vect_encoder]
        self.agent_pol_shared_modules = []

        self.extr_critics = nn.ModuleList(
            [nn.ModuleList([nn.Sequential(nn.Linear(action_size * (nagents - 1) + hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_size))
                            for _ in range(self.nagents)])
             for _ in range(self.n_intr_rew_heads + int(self.sep_extr_head))])

        self.intr_critics = nn.ModuleList(
            [nn.ModuleList([nn.Sequential(nn.Linear(action_size * (nagents - 1) + hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_size))
                            for _ in range(self.nagents)])
             for _ in range(self.n_intr_rew_heads)])

    def fully_shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.fully_shared_modules])

    def agent_pol_shared_parameters(self):
        """
        Parameters shared across agents
        """
        return chain(*[m.parameters() for m in self.agent_pol_shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.fully_shared_parameters():
            p.grad.data.mul_(1. / (self.nagents * (2 * self.n_intr_rew_heads + int(self.sep_extr_head))))

        for p in self.agent_pol_shared_parameters():
            p.grad.data.mul_(1. / (self.nagents * self.n_pol_heads))

    def forward(self, inps, agents=None, return_q=True, return_all_q=False):
        """
        Inputs:
            inps (list of PyTorch Matrices): Batch of state, obs, acs
                                             (latter 2 for each agent)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
        """
        if agents is None:
            agents = range(self.nagents)
        state_vect, obs, acs = inps
        # acs is always per agent; sometimes it is per agent per policy head
        # if we want different actions input to each corresponding critic head
        if not (isinstance(acs[0], tuple) or isinstance(acs[0], list)):
            acs = [[a] * self.n_pol_heads for a in acs]

        state_encoding = self.state_vect_encoder(state_vect)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            extr_all_qs = []
            extr_qs = []
            intr_all_qs = []
            intr_qs = []
            agent_rets = []
            for j in range(self.n_pol_heads):
                intr_rel_acs = [a[j] for k, a in enumerate(acs) if k != a_i]

                extr_critic_ins = torch.cat(intr_rel_acs + [state_encoding], dim=1)
                intr_critic_ins = extr_critic_ins

                int_acs = acs[a_i][j].max(dim=1, keepdim=True)[1]

                curr_all_q = self.extr_critics[j][a_i](extr_critic_ins)
                extr_all_qs.append(curr_all_q)
                extr_qs.append(curr_all_q.gather(1, int_acs))

                if (self.sep_extr_head and j > 0) or not self.sep_extr_head:
                    offset = -1 if self.sep_extr_head else 0
                    curr_all_q = self.intr_critics[j + offset][a_i](intr_critic_ins)
                    intr_all_qs.append(curr_all_q)
                    intr_qs.append(curr_all_q.gather(1, int_acs))

            if return_q:
                agent_rets.append((extr_qs, intr_qs))
            if return_all_q:
                agent_rets.append((extr_all_qs, intr_all_qs))
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
