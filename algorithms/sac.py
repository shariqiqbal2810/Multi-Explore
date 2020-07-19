import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, SGD
from gym import spaces
from itertools import chain
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients, pol_kl, RunningMeanStd, apply_to_all_elements
from utils.agents import Agent
from utils.policies import HeadSelector
from utils.critics import CentralCritic

MSELoss = torch.nn.MSELoss()
SmoothL1Loss = torch.nn.SmoothL1Loss()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

class SAC(object):
    """
    Training decentralized policies w/ centralized critic using
    (discrete action) Soft Actor-Critic
    """
    def __init__(self, nagents, obs_shape,
                 state_shape, action_size, gamma_e=0.95, gamma_i=0.95, tau=0.01,
                 hard_update_interval=None,
                 pi_lr=0.01, q_lr=0.01, phi_lr=0.1,
                 adam_eps=1e-8,
                 q_decay=1e-3, phi_decay=1e-4, reward_scale=10.,
                 head_reward_scale=25.,
                 pol_hidden_dim=64, critic_hidden_dim=64, nonlin=F.relu,
                 n_intr_rew_types=0,
                 sep_extr_head=False, beta=0.5, **kwargs):
        """
        Inputs:
            obs_shape (int): Dimensions of vector observations
            state_shape (int): Dimensions of vector global state
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = nagents
        n_pol_heads = n_intr_rew_types + int(sep_extr_head)

        self.agents = [Agent(obs_shape, action_size,
                             lr=pi_lr, adam_eps=adam_eps,
                             hidden_dim=pol_hidden_dim,
                             nonlin=nonlin,
                             n_pol_heads=n_pol_heads)
                       for _ in range(nagents)]

        self.critic = CentralCritic(state_shape[0], action_size,
                                    nagents, hidden_dim=critic_hidden_dim,
                                    n_intr_rew_heads=n_intr_rew_types,
                                    sep_extr_head=sep_extr_head)
        self.target_critic = CentralCritic(state_shape[0], action_size,
                                           nagents,
                                           hidden_dim=critic_hidden_dim,
                                           n_intr_rew_heads=n_intr_rew_types,
                                           sep_extr_head=sep_extr_head)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     eps=adam_eps, weight_decay=q_decay)

        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.head_reward_scale = head_reward_scale
        self.n_intr_rew_types = n_intr_rew_types
        self.sep_extr_head = sep_extr_head  # separate policy head only trained on extr rews
        self.n_pol_heads = n_pol_heads
        self.beta = beta
        self.head_selector = HeadSelector(nagents, n_pol_heads)
        self.head_selector_optimizer = SGD(self.head_selector.parameters(),
                                           lr=phi_lr, weight_decay=phi_decay)
        self.curr_pol_heads = self.sample_pol_heads()
        self.grad_norm_clip = 10
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def update_heads_onpol(self, mc_rets, ret_rms, soft=True, logger=None):
        """
        Update policy head selector(s) using monte carlo rollouts
        """
        curr_device = next(self.head_selector.parameters()).device
        heads = torch.tensor(self.curr_pol_heads[:1], device=curr_device).view(-1, 1)

        # get log_pi for all heads
        _, all_probs, all_log_probs, entropy = self.head_selector(
            return_all_probs=True, return_all_log_probs=True, return_entropy=True)
        all_probs = all_probs.flatten()
        log_pi = all_log_probs.gather(1, heads)

        q = torch.tensor(mc_rets.mean(), dtype=torch.float32,
                         device=curr_device)
        v = torch.tensor([rrms.mean for rrms in ret_rms], dtype=torch.float32,
                         device=curr_device)
        v = (v * all_probs).sum()

        if soft:
            loss = (-log_pi * ((-log_pi / self.head_reward_scale) + q - v).detach()).mean()
        else:
            loss = (-log_pi * (q - v).detach()).mean()
        self.head_selector_optimizer.zero_grad()
        loss.backward()
        self.head_selector_optimizer.step()
        if logger is not None:
            logger.add_scalar('head_entropy', entropy[0], self.niter)

            for i in range(self.n_pol_heads):
                logger.add_scalar('bandits/head%i' % i,
                                  all_probs[i], self.niter)

    def sample_pol_heads(self, uniform=False):
        """
        Sample new policy heads
        """
        if uniform:
            heads = [np.random.randint(self.n_pol_heads)]
        else:
            heads = self.head_selector().flatten().tolist()
        heads *= self.nagents
        return heads

    def set_pol_heads(self, pol_heads):
        """
        Set policy heads for rollouts
        """
        self.curr_pol_heads = pol_heads

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore, head=h)
                for a, obs, h in zip(self.agents, observations, self.curr_pol_heads)]

    def update_critic(self, sample, soft=True, logger=None, intr_rews=None,
                      **kwargs):
        """
        Update central critic for all agents
        """
        state, obs, acs, rews, next_state, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            pi_outs = pi(ob, return_log_pi=True)
            curr_next_ac, curr_next_log_pi = list(zip(*pi_outs))  # these contain entries per policy head
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = (next_state, next_obs, next_acs)
        critic_in = (state, obs, acs)
        next_qs = self.target_critic(trgt_critic_in)
        pred_qs = self.critic(critic_in)
        q_loss = 0

        for a_i, (neqs, niqs), log_pi, (peqs, piqs) in zip(
                range(self.nagents), next_qs, next_log_pis, pred_qs):
            if self.sep_extr_head:
                target_q = (rews.view(-1, 1) +
                            self.gamma_e * neqs[0] *
                            (1 - dones.view(-1, 1)))
                if soft:
                    target_q -= log_pi[0] / self.reward_scale
                q_loss += MSELoss(peqs[0], target_q.detach())
                neqs = neqs[1:]
                peqs = peqs[1:]
                log_pi = log_pi[1:]

            for i, neq, peq in zip(range(self.n_intr_rew_types), neqs, peqs):
                target_q = (rews.view(-1, 1) +
                            self.gamma_e * neq *
                            (1 - dones.view(-1, 1)))
                if soft:
                    target_q -= log_pi[i] / self.reward_scale
                q_loss += MSELoss(peq, target_q.detach())

            for i, niq, piq, in zip(range(self.n_intr_rew_types), niqs, piqs):
                target_q = (intr_rews[i][a_i].view(-1, 1) +
                            self.gamma_i * niq *
                            (1 - dones.view(-1, 1)))
                if soft:
                    target_q -= log_pi[i] / self.reward_scale
                q_loss += SmoothL1Loss(piq, target_q.detach())  # more robust to outliers

        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('grad_norms/critic', grad_norm, self.niter)
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None,
                        **kwargs):
        state, obs, acs, rews, next_state, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_probs = []
        all_log_pis = []
        all_pol_regs = []
        n_pol_heads = self.n_intr_rew_types + int(self.sep_extr_head)
        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            pi_outs = pi(
                ob, return_all_probs=True, return_all_log_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True, head=None)
            curr_ac, probs, log_probs, log_pi, pol_regs, ent = list(zip(*pi_outs))
            for j in range(n_pol_heads):
                logger.add_scalar('agent%i/pol%i_entropy' % (a_i, j), ent[j],
                                  self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_probs.append(log_probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = (state, obs, samp_acs)
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, ((eqs, iqs), (all_eqs, all_iqs)) in zip(
                range(self.nagents), all_probs, all_log_pis, all_pol_regs,
                      critic_rets):
            curr_agent = self.agents[a_i]
            pol_loss = 0.0
            if self.sep_extr_head:
                all_q = all_eqs[0]
                q = eqs[0]
                v = (all_q * probs[0]).sum(dim=1, keepdim=True)
                pol_target = q - v
                if soft:
                    pol_loss += (log_pi[0] * (log_pi[0] / self.reward_scale - pol_target).detach()).mean()
                else:
                    pol_loss += (log_pi[0] * (-pol_target).detach()).mean()
                for reg in pol_regs[0]:
                    pol_loss += 1e-3 * reg  # policy regularization
                eqs = eqs[1:]
                all_eqs = all_eqs[1:]
                probs = probs[1:]
                log_pi = log_pi[1:]
                pol_regs = pol_regs[1:]
            for j in range(self.n_intr_rew_types):
                q = eqs[j] + self.beta * iqs[j]
                all_q = all_eqs[j] + self.beta * all_iqs[j]
                v = (all_q * probs[j]).sum(dim=1, keepdim=True)
                pol_target = q - v
                if soft:
                    pol_loss += (log_pi[j] * (log_pi[j] / self.reward_scale - pol_target).detach()).mean()
                else:
                    pol_loss += (log_pi[j] * (-pol_target).detach()).mean()
                for reg in pol_regs[j]:
                    pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            curr_agent.policy.scale_shared_grads()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), self.grad_norm_clip/100.)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('grad_norms/agent%i_policy' % a_i,
                                  grad_norm, self.niter)
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        if self.hard_update_interval is None:
            soft_update(self.target_critic, self.critic, self.tau)
            for a in self.agents:
                soft_update(a.target_policy, a.policy, self.tau)
        elif self.niter % self.hard_update_interval == 0:
            hard_update(self.target_critic, self.critic)
            for a in self.agents:
                hard_update(a.target_policy, a.policy)


    def prep_training(self, device='cuda'):
        self.critic.train()
        self.target_critic.train()
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
            a.policy = a.policy.to(device)
            a.target_policy = a.target_policy.to(device)
        self.head_selector.train()
        self.head_selector.to(device)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.policy = a.policy.to(device)
        self.head_selector.eval()
        self.head_selector.to(device)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()},
                     'head_selector_params': {'head_selector': self.head_selector.state_dict(),
                                              'head_selector_optimizer': self.head_selector_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, nagents=2, gamma_e=0.95, gamma_i=0.95,
                      tau=0.01, hard_update_interval=None,
                      pi_lr=0.01, q_lr=0.01, phi_lr=0.1,
                      adam_eps=1e-8,
                      q_decay=1e-3, phi_decay=1e-4, reward_scale=10.,
                      head_reward_scale=25.,
                      pol_hidden_dim=128, critic_hidden_dim=128,
                      nonlin=F.relu, n_intr_rew_types=0, beta=0.5,
                      sep_extr_head=False, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        if type(env.observation_space) is spaces.Tuple:
            obs_shape = tuple(space.shape for space in
                              env.observation_space.spaces)
        else:
            obs_shape = env.observation_space.shape
        state_shape = env.state_space.shape
        action_size = env.action_space.n

        init_dict = {'nagents': nagents,
                     'gamma_e': gamma_e, 'gamma_i': gamma_i, 'tau': tau,
                     'hard_update_interval': hard_update_interval,
                     'pi_lr': pi_lr, 'q_lr': q_lr, 'phi_lr': phi_lr,
                     'adam_eps': adam_eps,
                     'q_decay': q_decay, 'phi_decay': phi_decay,
                     'reward_scale': reward_scale,
                     'head_reward_scale': head_reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'nonlin': nonlin,
                     'n_intr_rew_types': n_intr_rew_types,
                     'sep_extr_head': sep_extr_head,
                     'beta': beta,
                     'obs_shape': obs_shape,
                     'state_shape': state_shape,
                     'action_size': action_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False, load_ir=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params, load_ir=load_ir)

        head_selector_params = save_dict['head_selector_params']
        instance.head_selector.load_state_dict(head_selector_params['head_selector'])
        instance.head_selector_optimizer.load_state_dict(head_selector_params['head_selector_optimizer'])
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
