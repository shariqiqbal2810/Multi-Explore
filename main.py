import argparse
import torch
import os
import multiprocessing
import numpy as np
from vizdoom import ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException
from gym.spaces import Box, Discrete
from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv
from utils.misc import apply_to_all_elements, timeout, RunningMeanStd
from algorithms.sac import SAC
from envs.ma_vizdoom.ma_vizdoom import VizdoomMultiAgentEnv
from envs.magw.multiagent_env import GridWorld, VectObsEnv

AGENT_CMAPS = ['Reds', 'Blues', 'Greens', 'Wistia']

def get_count_based_novelties(env, state_inds, device='cpu'):
    env_visit_counts = env.get_visit_counts()

    # samp_visit_counts[i,j,k] is # of times agent j has visited the state that agent k occupies at time i
    samp_visit_counts = np.concatenate(
        [np.concatenate(
            [env_visit_counts[j][tuple(zip(*state_inds[k]))].reshape(-1, 1, 1)
             for j in range(config.num_agents)], axis=1)
         for k in range(config.num_agents)], axis=2)

    # how novel each agent considers all agents observations at every step
    novelties = np.power(np.maximum(samp_visit_counts, 1), -config.decay)
    return torch.tensor(novelties, device=device, dtype=torch.float32)

def get_intrinsic_rewards(novelties, config, intr_rew_rms,
                          update_irrms=False, active_envs=None, device='cpu'):
    if update_irrms:
        assert active_envs is not None
    intr_rews = []

    for i, exp_type in enumerate(config.explr_types):
        if exp_type == 0:  # independent
            intr_rews.append([novelties[:, ai, ai] for ai in range(config.num_agents)])
        elif exp_type == 1:  # min
            intr_rews.append([novelties[:, :, ai].min(axis=1)[0] for ai in range(config.num_agents)])
        elif exp_type == 2:  # covering
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 3:  # burrowing
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] = 0.0
                rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 4:  # leader-follow
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                if ai == 0:
                    rew[rew > 0.0] = 0.0
                    rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                else:
                    rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                    rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)

    for i in range(len(config.explr_types)):
        for j in range(config.num_agents):
            if update_irrms:
                intr_rew_rms[i][j].update(intr_rews[i][j].cpu().numpy(), active_envs=active_envs)
            intr_rews[i][j] = intr_rews[i][j].to(device)
            norm_fac = torch.tensor(np.sqrt(intr_rew_rms[i][j].var),
                                    device=device, dtype=torch.float32)
            intr_rews[i][j] /= norm_fac

    return intr_rews

def make_parallel_env(config, seed):
    lock = multiprocessing.Lock()
    def get_env_fn(rank):
        def init_env():
            if config.env_type == 'gridworld':
                env = VectObsEnv(GridWorld([config.map_ind],
                                           task_length=1, train_combos=[(0, 0)],
                                           test_combos=[(0, 0)],
                                           seed=(seed * 1000 + rank),
                                           task_config=config.task_config,
                                           num_agents=config.num_agents,
                                           need_get=False,
                                           stay_act=True), l=3)
            else:  # vizdoom
                env = VizdoomMultiAgentEnv(task_id=config.task_config,
                                           env_id=(seed - 1) * 64 + rank,  # assumes no more than 64 environments per run
                                           seed=seed * 640 + rank * 10,  # assumes no more than 10 agents per run
                                           lock=lock,
                                           skip_frames=config.frame_skip)
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(i) for i in
                          range(config.n_rollout_threads)])


def run(config):
    torch.set_num_threads(1)
    env_descr = 'map%i_%iagents_task%i' % (config.map_ind, config.num_agents,
                                           config.task_config)
    model_dir = Path('./models') / config.env_type / env_descr / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config, run_num)
    if config.nonlinearity == 'relu':
        nonlin = torch.nn.functional.relu
    elif config.nonlinearity == 'leaky_relu':
        nonlin = torch.nn.functional.leaky_relu
    if config.intrinsic_reward == 0:
        n_intr_rew_types = 0
        sep_extr_head = True
    else:
        n_intr_rew_types = len(config.explr_types)
        sep_extr_head = False
    n_rew_heads = n_intr_rew_types + int(sep_extr_head)

    model = SAC.init_from_env(env,
                              nagents=config.num_agents,
                              tau=config.tau,
                              hard_update_interval=config.hard_update,
                              pi_lr=config.pi_lr,
                              q_lr=config.q_lr,
                              phi_lr=config.phi_lr,
                              adam_eps=config.adam_eps,
                              q_decay=config.q_decay,
                              phi_decay=config.phi_decay,
                              gamma_e=config.gamma_e,
                              gamma_i=config.gamma_i,
                              pol_hidden_dim=config.pol_hidden_dim,
                              critic_hidden_dim=config.critic_hidden_dim,
                              nonlin=nonlin,
                              reward_scale=config.reward_scale,
                              head_reward_scale=config.head_reward_scale,
                              beta=config.beta,
                              n_intr_rew_types=n_intr_rew_types,
                              sep_extr_head=sep_extr_head)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 env.state_space,
                                 env.observation_space,
                                 env.action_space)
    intr_rew_rms = [[RunningMeanStd()
                     for i in range(config.num_agents)]
                    for j in range(n_intr_rew_types)]
    eps_this_turn = 0  # episodes so far this turn
    active_envs = np.ones(config.n_rollout_threads)  # binary indicator of whether env is active
    env_times = np.zeros(config.n_rollout_threads, dtype=int)
    env_ep_extr_rews = np.zeros(config.n_rollout_threads)
    env_extr_rets = np.zeros(config.n_rollout_threads)
    env_ep_intr_rews = [[np.zeros(config.n_rollout_threads) for i in range(config.num_agents)]
                        for j in range(n_intr_rew_types)]
    recent_ep_extr_rews = deque(maxlen=100)
    recent_ep_intr_rews = [[deque(maxlen=100) for i in range(config.num_agents)]
                           for j in range(n_intr_rew_types)]
    recent_ep_lens = deque(maxlen=100)
    recent_found_treasures = [deque(maxlen=100) for i in range(config.num_agents)]
    meta_turn_rets = []
    extr_ret_rms = [RunningMeanStd() for i in range(n_rew_heads)]
    t = 0
    steps_since_update = 0

    state, obs = env.reset()

    while t < config.train_time:
        model.prep_rollouts(device='cuda' if config.gpu_rollout else 'cpu')
        # convert to torch tensor
        torch_obs = apply_to_all_elements(obs, lambda x: torch.tensor(x, dtype=torch.float32, device='cuda' if config.gpu_rollout else 'cpu'))
        # get actions as torch tensors
        torch_agent_actions = model.step(torch_obs, explore=True)
        # convert actions to numpy arrays
        agent_actions = apply_to_all_elements(torch_agent_actions, lambda x: x.cpu().data.numpy())
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(int(active_envs.sum()))]
        try:
            with timeout(seconds=1):
                next_state, next_obs, rewards, dones, infos = env.step(actions, env_mask=active_envs)
        # either environment got stuck or vizdoom crashed (vizdoom is unstable w/ multi-agent scenarios)
        except (TimeoutError, ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException) as e:
            print("Environments are broken...")
            env.close(force=True)
            print("Closed environments, starting new...")
            env = make_parallel_env(config, run_num)
            state, obs = env.reset()
            env_ep_extr_rews[active_envs.astype(bool)] = 0.0
            env_extr_rets[active_envs.astype(bool)] = 0.0
            for i in range(n_intr_rew_types):
                for j in range(config.num_agents):
                    env_ep_intr_rews[i][j][active_envs.astype(bool)] = 0.0
            env_times = np.zeros(config.n_rollout_threads, dtype=int)
            state = apply_to_all_elements(state, lambda x: x[active_envs.astype(bool)])
            obs = apply_to_all_elements(obs, lambda x: x[active_envs.astype(bool)])
            continue

        steps_since_update += int(active_envs.sum())
        if config.intrinsic_reward == 1:
            # if using state-visit counts, store state indices
            # shape = (n_envs, n_agents, n_inds)
            state_inds = np.array([i['visit_count_lookup'] for i in infos],
                                  dtype=int)
            state_inds_t = state_inds.transpose(1, 0, 2)
            novelties = get_count_based_novelties(env, state_inds_t, device='cpu')
            intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                              update_irrms=True, active_envs=active_envs,
                                              device='cpu')
            intr_rews = apply_to_all_elements(intr_rews, lambda x: x.numpy().flatten())
        else:
            intr_rews = None
            state_inds = None
            state_inds_t = None

        replay_buffer.push(state, obs, agent_actions, rewards, next_state, next_obs, dones,
                           state_inds=state_inds)
        env_ep_extr_rews[active_envs.astype(bool)] += np.array(rewards)
        env_extr_rets[active_envs.astype(bool)] += np.array(rewards) * config.gamma_e**(env_times[active_envs.astype(bool)])
        env_times += active_envs.astype(int)
        if intr_rews is not None:
            for i in range(n_intr_rew_types):
                for j in range(config.num_agents):
                    env_ep_intr_rews[i][j][active_envs.astype(bool)] += intr_rews[i][j]
        over_time = env_times >= config.max_episode_length
        full_dones = np.zeros(config.n_rollout_threads)
        for i, env_i in enumerate(np.where(active_envs)[0]):
            full_dones[env_i] = dones[i]
        need_reset = np.logical_or(full_dones, over_time)
        # create masks ONLY for active envs
        active_over_time = env_times[active_envs.astype(bool)] >= config.max_episode_length
        active_need_reset = np.logical_or(dones, active_over_time)
        if any(need_reset):
            try:
                with timeout(seconds=1):
                    # reset any environments that are past the max number of time steps or done
                    state, obs = env.reset(need_reset=need_reset)
            # either environment got stuck or vizdoom crashed (vizdoom is unstable w/ multi-agent scenarios)
            except (TimeoutError, ViZDoomErrorException, ViZDoomIsNotRunningException, ViZDoomUnexpectedExitException) as e:
                print("Environments are broken...")
                env.close(force=True)
                print("Closed environments, starting new...")
                env = make_parallel_env(config, run_num)
                state, obs = env.reset()
                # other envs that were force reset (rest taken care of in subsequent code)
                other_reset = np.logical_not(need_reset)
                env_ep_extr_rews[other_reset.astype(bool)] = 0.0
                env_extr_rets[other_reset.astype(bool)] = 0.0
                for i in range(n_intr_rew_types):
                    for j in range(config.num_agents):
                        env_ep_intr_rews[i][j][other_reset.astype(bool)] = 0.0
                env_times = np.zeros(config.n_rollout_threads, dtype=int)
        else:
            state, obs = next_state, next_obs
        for env_i in np.where(need_reset)[0]:
            recent_ep_extr_rews.append(env_ep_extr_rews[env_i])
            meta_turn_rets.append(env_extr_rets[env_i])
            if intr_rews is not None:
                for j in range(n_intr_rew_types):
                    for k in range(config.num_agents):
                        # record intrinsic rewards per step (so we don't confuse shorter episodes with less intrinsic rewards)
                        recent_ep_intr_rews[j][k].append(env_ep_intr_rews[j][k][env_i] / env_times[env_i])
                        env_ep_intr_rews[j][k][env_i] = 0

            recent_ep_lens.append(env_times[env_i])
            env_times[env_i] = 0
            env_ep_extr_rews[env_i] = 0
            env_extr_rets[env_i] = 0
            eps_this_turn += 1

            if eps_this_turn + active_envs.sum() - 1 >= config.metapol_episodes:
                active_envs[env_i] = 0

        for i in np.where(active_need_reset)[0]:
            for j in range(config.num_agents):
                # len(infos) = number of active envs
                recent_found_treasures[j].append(infos[i]['n_found_treasures'][j])

        if eps_this_turn >= config.metapol_episodes:
            if not config.uniform_heads and n_rew_heads > 1:
                meta_turn_rets = np.array(meta_turn_rets)
                if all(errms.count < 1 for errms in extr_ret_rms):
                    for errms in extr_ret_rms:
                        errms.mean = meta_turn_rets.mean()
                extr_ret_rms[model.curr_pol_heads[0]].update(meta_turn_rets)
                for i in range(config.metapol_updates):
                    model.update_heads_onpol(meta_turn_rets, extr_ret_rms, logger=logger)
            pol_heads = model.sample_pol_heads(uniform=config.uniform_heads)
            model.set_pol_heads(pol_heads)
            eps_this_turn = 0
            meta_turn_rets = []
            active_envs = np.ones(config.n_rollout_threads)

        if any(need_reset):  # reset returns state and obs for all envs, so make sure we're only looking at active
            state = apply_to_all_elements(state, lambda x: x[active_envs.astype(bool)])
            obs = apply_to_all_elements(obs, lambda x: x[active_envs.astype(bool)])

        if (len(replay_buffer) >= max(config.batch_size,
                                      config.steps_before_update) and
                (steps_since_update >= config.steps_per_update)):
            steps_since_update = 0
            print('Updating at time step %i' % t)
            model.prep_training(device='cuda' if config.use_gpu else 'cpu')

            for u_i in range(config.num_updates):
                sample = replay_buffer.sample(config.batch_size,
                                              to_gpu=config.use_gpu,
                                              state_inds=(config.intrinsic_reward == 1))

                if config.intrinsic_reward == 0:  # no intrinsic reward
                    intr_rews = None
                    state_inds = None
                else:
                    sample, state_inds = sample
                    novelties = get_count_based_novelties(
                        env, state_inds,
                        device='cuda' if config.use_gpu else 'cpu')
                    intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                                      update_irrms=False,
                                                      device='cuda' if config.use_gpu else 'cpu')

                model.update_critic(sample, logger=logger, intr_rews=intr_rews)
                model.update_policies(sample, logger=logger)
                model.update_all_targets()
            if len(recent_ep_extr_rews) > 10:
                logger.add_scalar('episode_rewards/extrinsic/mean',
                                  np.mean(recent_ep_extr_rews), t)
                logger.add_scalar('episode_lengths/mean',
                                  np.mean(recent_ep_lens), t)
                if config.intrinsic_reward == 1:
                    for i in range(n_intr_rew_types):
                        for j in range(config.num_agents):
                            logger.add_scalar('episode_rewards/intrinsic%i_agent%i/mean' % (i, j),
                                              np.mean(recent_ep_intr_rews[i][j]), t)
                for i in range(config.num_agents):
                    logger.add_scalar('agent%i/n_found_treasures' % i, np.mean(recent_found_treasures[i]), t)
                logger.add_scalar('total_n_found_treasures', sum(np.array(recent_found_treasures[i]) for i in range(config.num_agents)).mean(), t)

        if t % config.save_interval < config.n_rollout_threads:
            model.prep_training(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_%isteps.pt' % (t + 1)))
            model.save(run_dir / 'model.pt')

        t += active_envs.sum()
    model.prep_training(device='cpu')
    model.save(run_dir / 'model.pt')
    logger.close()
    env.close(force=(config.env_type == 'vizdoom'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--env_type", type=str, default='gridworld', choices=['gridworld', 'vizdoom'])
    parser.add_argument("--map_ind", help="Index of map to use (only for gridworld)", type=int,
                        default=1)
    parser.add_argument("--num_agents", help="Number of agents (vizdoom only supports 2 at the moment)", type=int,
                        default=2)
    parser.add_argument("--task_config", help="Index of task configuration",
                        type=int, default=1)
    parser.add_argument("--frame_skip", help="How many frames to skip per step (only for vizdoom)", type=int,
                        default=2)
    parser.add_argument("--intrinsic_reward", type=int, default=1,
                        help="Use intrinsic reward for exploration\n" +
                             "0: No intrinsic reward\n" +
                             "1 (default): Intrinsic reward using state visit counts")
    parser.add_argument("--explr_types", type=int, nargs='*', default=[0, 1, 2, 3, 4],
                        help="Type of exploration, can provide multiple\n" +
                             "0: Independent exploration\n" +
                             "1: Minimum exploration\n" +
                             "2: Covering exploration\n" +
                             "3: Burrowing exploration\n" +
                             "4: Leader-Follower exploration\n")
    parser.add_argument("--uniform_heads", action="store_true",
                        help="Meta-policy samples all heads uniformly")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weighting for intrinsic reward")
    parser.add_argument("--decay", type=float, default=0.7,
                        help="Decay rate for state-visit counts in intrinsic reward")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int,
                        help="Set to 5e5 for ViZDoom (if memory limited)")
    parser.add_argument("--train_time", default=int(1e6), type=int)
    parser.add_argument("--max_episode_length", default=500, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--metapol_episodes", default=12, type=int,
                        help="Number of episodes to rollout before updating the meta-policy " +
                             "(policy selector). Better if a multiple of n_rollout_threads")
    parser.add_argument("--steps_before_update", default=0, type=int)
    parser.add_argument("--num_updates", default=50, type=int,
                        help="Number of SAC updates per cycle")
    parser.add_argument("--metapol_updates", default=100, type=int,
                        help="Number of updates for meta-policy per turn")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training. \n"
                             "Set to 128 for ViZDoom scenarios")
    parser.add_argument("--save_interval", default=100000, type=int)
    parser.add_argument("--pol_hidden_dim", default=32, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int,
                        help="Set to 256 for ViZDoom scenarios")
    parser.add_argument("--nonlinearity", default="relu", type=str,
                        choices=["relu", "leaky_relu"])
    parser.add_argument("--pi_lr", default=0.001, type=float,
                        help="Set to 0.0005 for ViZDoom scenarios")
    parser.add_argument("--q_lr", default=0.001, type=float,
                        help="Set to 0.0005 for ViZDoom scenarios")
    parser.add_argument("--phi_lr", default=0.04, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--q_decay", default=1e-3, type=float)
    parser.add_argument("--phi_decay", default=1e-3, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--hard_update", default=None, type=int,
                        help="If specified, use hard update for target critic" +
                             "every _ steps instead of soft update w/ tau")
    parser.add_argument("--gamma_e", default=0.99, type=float)
    parser.add_argument("--gamma_i", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--head_reward_scale", default=5., type=float)
    parser.add_argument("--use_gpu", action='store_true',
                        help='Use GPU for training')
    parser.add_argument("--gpu_rollout", action='store_true',
                        help='Use GPU for rollouts (more useful for lots of '
                        'parallel envs or image-based observations')

    config = parser.parse_args()

    run(config)
