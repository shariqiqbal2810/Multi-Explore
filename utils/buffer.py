import numpy as np
import torch
from gym import spaces

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts, global state,
    local observations, and shared rewards
    """
    def __init__(self, max_steps, num_agents, state_space, observation_space, action_space):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.state_vect_buff = np.zeros((max_steps, state_space.shape[0]), dtype=np.float32)
        self.next_state_vect_buff = np.zeros((max_steps, state_space.shape[0]), dtype=np.float32)
        self.rew_buff = np.zeros(max_steps, dtype=np.float32)
        self.done_buff = np.zeros(max_steps, dtype=np.uint8)
        self.obs_buffs = []
        self.ac_buffs = []
        self.next_obs_buffs = []
        self.state_inds_buffs = []  # stores x,y coords of agent state (for count based exploration methods)

        for i in range(self.num_agents):
            if type(observation_space) is spaces.Tuple:
                self.mult_obs = True
                agent_obs_buff = [np.zeros((max_steps, *space.shape), dtype=space.dtype) for space in observation_space.spaces]
                self.obs_buffs.append(agent_obs_buff)
                agent_next_obs_buff = [np.zeros((max_steps, *space.shape), dtype=space.dtype) for space in observation_space.spaces]
                self.next_obs_buffs.append(agent_next_obs_buff)
            else:
                self.mult_obs = False
                agent_obs_buff = np.zeros((max_steps, *observation_space.shape), dtype=observation_space.dtype)
                self.obs_buffs.append(agent_obs_buff)
                agent_next_obs_buff = np.zeros((max_steps, *observation_space.shape), dtype=observation_space.dtype)
                self.next_obs_buffs.append(agent_next_obs_buff)
            self.ac_buffs.append(np.zeros((max_steps, action_space.n), dtype=np.uint8))
            self.state_inds_buffs.append(np.zeros((max_steps, 2), dtype=int))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, states, observations, actions, rewards, next_states,
             next_observations, dones, state_inds=None):
        nentries = states.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.state_vect_buff = np.roll(self.state_vect_buff, rollover, axis=0)
            self.next_state_vect_buff = np.roll(self.next_state_vect_buff, rollover, axis=0)
            self.rew_buff = np.roll(self.rew_buff, rollover)
            self.done_buff = np.roll(self.done_buff, rollover)
            for ai in range(self.num_agents):
                if self.mult_obs:
                    for oi in range(len(self.obs_buffs[ai])):
                        self.obs_buffs[ai][oi] = np.roll(self.obs_buffs[ai][oi],
                                                              rollover, axis=0)
                        self.next_obs_buffs[ai][oi] = np.roll(self.next_obs_buffs[ai][oi],
                                                               rollover, axis=0)
                else:
                    self.obs_buffs[ai] = np.roll(self.obs_buffs[ai],
                                                          rollover, axis=0)
                    self.next_obs_buffs[ai] = np.roll(self.next_obs_buffs[ai],
                                                           rollover, axis=0)
                self.ac_buffs[ai] = np.roll(self.ac_buffs[ai],
                                                 rollover, axis=0)
                self.state_inds_buffs[ai] = np.roll(self.state_inds_buffs[ai],
                                                    rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_steps
        state_vect = states
        next_state_vect = next_states
        fill_inds = slice(self.curr_i, self.curr_i + nentries)
        self.state_vect_buff[fill_inds] = state_vect
        self.rew_buff[fill_inds] = rewards
        self.done_buff[fill_inds] = dones
        self.next_state_vect_buff[fill_inds] = next_state_vect
        for ai in range(self.num_agents):
            obs = observations[ai]
            next_obs = next_observations[ai]
            if self.mult_obs:
                for oi in range(len(self.obs_buffs[ai])):
                    self.obs_buffs[ai][oi][fill_inds] = obs[oi]
                    self.next_obs_buffs[ai][oi][fill_inds] = next_obs[oi]
            else:
                self.obs_buffs[ai][fill_inds] = obs
                self.next_obs_buffs[ai][fill_inds] = next_obs
            self.ac_buffs[ai][fill_inds] = actions[ai]
            if state_inds is not None:
                self.state_inds_buffs[ai][fill_inds] = state_inds[:, ai, :]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False, state_inds=False, recent=None):
        if recent is not None:
            if self.filled_i != self.max_steps:
                min_ind = max(0, self.curr_i - recent)
            else:
                min_ind = self.curr_i - recent
            poss_inds = np.arange(min_ind, self.curr_i)
            if N != recent:
                inds = np.random.choice(poss_inds, size=N)
            else:
                inds = poss_inds
        else:
            inds = np.random.randint(0, self.filled_i, size=N)
        cast = lambda x: torch.tensor(x, device='cuda' if to_gpu else 'cpu',
                                      dtype=torch.float32)
        if norm_rews:
            # self.rew_mean = self.rew_buff[:self.filled_i].mean()
            self.rew_std = self.rew_buff[:self.filled_i].std()
            if not np.isclose(self.rew_std, 0.0):
                ret_rews = cast((self.rew_buff[inds]) / self.rew_std)
            else:
                ret_rews = cast(self.rew_buff[inds])
        else:
            ret_rews = cast(self.rew_buff[inds])
        # state, obs, acs, rews, next_state, next_obs, dones
        if self.mult_obs:
            ret_obs = [tuple(cast(self.obs_buffs[ai][oi][inds]) for oi in range(len(self.obs_buffs[ai]))) for ai in range(self.num_agents)]
            ret_next_obs = [tuple(cast(self.next_obs_buffs[ai][oi][inds]) for oi in range(len(self.next_obs_buffs[ai]))) for ai in range(self.num_agents)]
        else:
            ret_obs = [cast(self.obs_buffs[ai][inds]) for ai in range(self.num_agents)]
            ret_next_obs = [cast(self.next_obs_buffs[ai][inds]) for ai in range(self.num_agents)]
        rets = (cast(self.state_vect_buff[inds]),
                ret_obs,
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                cast(self.next_state_vect_buff[inds]),
                ret_next_obs,
                cast(self.done_buff[inds]))
        if state_inds:
            return (rets, [self.state_inds_buffs[i][inds] for i in range(self.num_agents)])
        else:
            return rets
