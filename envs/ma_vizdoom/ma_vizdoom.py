# Modified from https://gist.github.com/alex-petrenko/5cf4686e6494ad3260c87f00d27b7e49
# and https://github.com/alex-petrenko/vizdoomgym/blob/master/vizdoomgym/envs/vizdoomenv.py
import logging
import psutil
import os
import multiprocessing
from time import sleep
from enum import Enum
from multiprocessing import JoinableQueue, Process
from queue import Empty

import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from vizdoom import *

from .wrappers import StackFramesWrapper, ResizeAndGrayscaleWrapper, ResizeWrapper

log = logging.getLogger(__name__)

CONFIGS = [ # config_filename, num_actions, (x_min, y_min, x_max, y_max)
    ['my_way_home_multi_easy.cfg', 3, (160, -704, 1120, 128)],  # 0
    ['my_way_home_multi_task1.cfg', 3, (160, -704, 1120, 128)],  # 1
    ['my_way_home_multi_task2.cfg', 3, (160, -704, 1120, 128)],  # 2
    ['my_way_home_multi_task3.cfg', 3, (160, -704, 1120, 128)],  # 3
]

RESIZE_W = RESIZE_H = 48  # instead of 42, so that 4 conv layers w/ stride 2 fits more nicely
MAX_VEL = 7.5  # approximately...

def make_component_env(level=0, player_id=-1, port=5029, num_players=2, skip_frames=4, stack_frames=False, grayscale=False):
    """
    Build the env for one agent
    """
    env = VizdoomEnvMultiplayer(level, player_id, port, num_players, skip_frames)
    # add wrappers
    if grayscale:
        env = ResizeAndGrayscaleWrapper(env, RESIZE_W, RESIZE_H)
    else:
        env = ResizeWrapper(env, RESIZE_W, RESIZE_H)
    if stack_frames:
        env = StackFramesWrapper(env, stack_past_frames=skip_frames)
    return env


class VizdoomEnv(gym.Env):

    def __init__(self,
                 level,
                 show_automap=False,
                 skip_frames=1,
                 level_map='map01'):
        self.initialized = False

        # init game
        self.level = level
        self.show_automap = show_automap
        self.coord_limits = CONFIGS[self.level][2]
        self.skip_frames = skip_frames
        self.game = None
        self.state = None

        self.curr_seed = 0
        self.level_map = level_map

        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.action_space = spaces.Discrete(CONFIGS[self.level][1])

        self.viewer = None

        self.seed()

    def calc_observation_space(self):
        self.observation_space = spaces.Box(0, 255, (self.screen_w, self.screen_h, self.channels), dtype=np.uint8)

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[self.level][0]))
        self.game.set_screen_resolution(self.screen_resolution)
        # Setting an invalid level map will cause the game to freeze silently
        self.game.set_doom_map(self.level_map)
        self.game.set_seed(self.rng.random_integers(0, 2**32-1))

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human':
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
        else:
            raise Exception('Unsupported mode')

        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = 'ffffff'
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self.game.init()

        self.initialized = True

    def _start_episode(self):
        # # TODO: Why set the seed here? Game is already initialized.
        # if self.curr_seed > 0:
        #     self.game.set_seed(self.curr_seed)
        #     self.curr_seed = self.rng.random_integers(0, 2**32 - 1)
        self.game.new_episode()
        return

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed, max_bytes=4)
        self.rng, _ = seeding.np_random(seed=self.curr_seed)
        return [self.curr_seed, self.rng]

    def step(self, action):
        self._ensure_initialized()
        info = {'num_frames': self.skip_frames}

        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            info.update(self._get_info())
            self._update_histogram(info)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return observation, reward, done, info

    def reset(self, mode='algo'):
        self._ensure_initialized(mode)

        self._start_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer

        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])  # bgr to rgb

            h, w = img.shape[:2]
            render_w = 640

            if w < render_w:
                render_h = int(640 * h / w)
                img = cv2.resize(img, (render_w, render_h))

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer(maxwidth=800)
            self.viewer.imshow(img)
        except AttributeError:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def play_human_mode(self, num_episodes=3):
        for episode in range(num_episodes):
            self.reset('human')
            while not self.game.is_episode_finished():
                self.game.advance_action()
                state = self.game.get_state()
                total_reward = self.game.get_total_reward()

                if state is not None:
                    print('===============================')
                    print('State: #' + str(state.number))
                    print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
                    print('Reward: \t' + str(self.game.get_last_reward()))
                    print('Total Reward: \t' + str(total_reward))

                    if self.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow('ViZDoom Automap Buffer', map_)
                        cv2.waitKey(28)
                    else:
                        sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
            if self.show_automap:
                cv2.destroyAllWindows()

        sleep(1)
        print('===============================')
        print('Done')

    def _get_info(self):
        return {'pos': self.get_positions()}

    def get_info_all(self):
        info = self._get_info()
        return info

    def get_positions(self):
        return self._get_positions(self.game.get_state().game_variables)

    def _get_positions(self, variables):
        coords = [np.nan] * 4
        if len(variables) >= 4:
            coords = variables

        return {'agent_x': coords[1], 'agent_y': coords[2], 'agent_a': coords[3]}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info['pos']['agent_x'], info['pos']['agent_y']

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        self.current_histogram[dx, dy] += 1


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(self, level, player_id, port, num_players, skip_frames, level_map='map01',
                 bin_resolution=32):
        super().__init__(level, skip_frames=skip_frames, level_map=level_map)

        self.port = port
        self.player_id = player_id
        self.num_players = num_players
        self.bin_resolution = bin_resolution
        self.timestep = 0
        self.reward = 0.0
        self.update_state = True

        # Histogram to track positional coverage
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            X = (self.coord_limits[2] - self.coord_limits[0])
            Y = (self.coord_limits[3] - self.coord_limits[1])
            self.x_bins = int(X / self.bin_resolution)
            self.y_bins = int(Y / self.bin_resolution)
            self.current_histogram = np.zeros((self.x_bins, self.y_bins), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)
            self.visit_counts = np.zeros((self.x_bins, self.y_bins), dtype=np.int32)

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()

        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[self.level][0]))
        live_rew = self.game.get_living_reward()
        self.game.set_living_reward(live_rew / self.skip_frames)  # make living reward consistent regardless of skip_frames
        if mode == 'algo':
            self.screen_w, self.screen_h, self.channels = 160, 120, 3
            self.screen_resolution = ScreenResolution.RES_160X120
        elif mode == 'watch':
            self.screen_w, self.screen_h, self.channels = 400, 300, 3
            self.screen_resolution = ScreenResolution.RES_400X300
        self.calc_observation_space()
        self.game.set_screen_resolution(self.screen_resolution)
        # Setting an invalid level map will cause the game to freeze silently
        # self.game.set_doom_map(self.level_map)
        self.game.set_seed(self.rng.random_integers(0, 2**32-1))

        self.game.set_window_visible(False)

        if self._is_server():
            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            self.game.add_game_args(
                f'-host {self.num_players} '
                '-netmode 0 '
                f'-port {self.port} '
            )
        else:
            # TODO: name
            # Join existing game.
            self.game.add_game_args(f'-join 127.0.0.1:{self.port}')  # Connect to a host for a multiplayer game.

        # Name your agent
        self.game.add_game_args(f'+name Player{self.player_id}')

        self.game.set_mode(Mode.PLAYER)

        self.game.init()

        self.initialized = True

    def reset(self, mode='algo'):
        self._ensure_initialized(mode)
        self.timestep = 0
        self.full_step = True
        self.game.new_episode()

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self.state = self.game.get_state()
        img = self.state.screen_buffer
        return np.transpose(img, (1, 2, 0))

    def _get_info(self, eps=1e-8):
        x_pos = (self.game.get_game_variable(GameVariable.POSITION_X) - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        y_pos = (self.game.get_game_variable(GameVariable.POSITION_Y) - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])
        x_vel = self.game.get_game_variable(GameVariable.VELOCITY_X) / 7.5
        y_vel = self.game.get_game_variable(GameVariable.VELOCITY_Y) / 7.5
        orient = (self.game.get_game_variable(GameVariable.CAMERA_ANGLE) / 360) * 2 * np.pi
        # represent orientation as sin and cos, so there are no discontinuities (since 0 and 360 should be represented as similar)
        sin_orient = np.sin(orient)
        cos_orient = np.cos(orient)
        item1_coll = int(self.game.get_game_variable(GameVariable.USER1))  # who has collected item 1
        item2_coll = int(self.game.get_game_variable(GameVariable.USER2))
        if item1_coll == 2:
            item1_onehot = np.ones(2)
        else:
            item1_onehot = np.zeros(2)
            if item1_coll != -1:
                item1_onehot[item1_coll] = 1
        if item2_coll == 2:
            item2_onehot = np.ones(2)
        else:
            item2_onehot = np.zeros(2)
            if item2_coll != -1:
                item2_onehot[item2_coll] = 1

        n_found_treasures = int(item1_coll in (self.player_id, 2)) + int(item2_coll in (self.player_id, 2))

        # Convert normalized coordinates to histogram indices
        # Subtract eps to exclude upper bound of dx, dy
        x_ind = int((x_pos - eps) * self.x_bins)
        y_ind = int((y_pos - eps) * self.y_bins)

        # one-hot x,y coordinates
        x_onehot = np.zeros(self.x_bins)
        x_onehot[x_ind] = 1
        y_onehot = np.zeros(self.y_bins)
        y_onehot[y_ind] = 1

        return {'pos': (x_pos, y_pos),
                'vel': (x_vel, y_vel),
                'orient': (sin_orient, cos_orient),
                'item1': item1_onehot,
                'item2': item2_onehot,
                'hist_inds': [x_ind, y_ind],
                'onehot_pos': np.concatenate((x_onehot, y_onehot)),
                'n_found_treasures': n_found_treasures}

    def _update_histogram(self, info):
        if self.current_histogram is None:
            return
        x_ind, y_ind = info['hist_inds']

        self.current_histogram[x_ind, y_ind] += 1
        self.visit_counts[x_ind, y_ind] += 1

    def step(self, action):
        self._ensure_initialized()
        info = {'num_frames': self.skip_frames}

        if type(action) is int:
            # convert action to vizdoom action space (one hot)
            act = np.zeros(self.action_space.n)
            act[action] = 1
            act = np.uint8(act)
            act = act.tolist()
        elif type(action) is np.ndarray:
            act = action.tolist()

        self.game.set_action(act)
        # always update state (even on sub-step) bc we may want intermediate frames for frame stacking
        self.game.advance_action(1, True)
        self.reward += self.game.get_last_reward()
        self.timestep += 1

        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            info.update(self._get_info())
            self._update_histogram(info)
            info.update({'visit_counts': self.visit_counts})
            self.last_info = info
            self.last_obs = observation
        else:
            info = self.last_info
            observation = self.last_obs
        reward = self.reward
        if self.full_step:
            self.reward = 0.0  # reset to accumulate on next sub-steps
        return observation, reward, done, info

    def get_obs(self):
        if not self.game.is_episode_finished():
            state = self.game.get_state()
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = self.last_obs
        return observation

def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.exception(msg)


class TaskType(Enum):
    INIT, TERMINATE, RESET, SUB_STEP, STEP, INFO, RENDER, GET_OBS = range(8)


class MultiAgentEnvWorker:
    def __init__(self, task_id, player_id, num_players, make_env_func, seed, skip_frames,
                 stack_frames, grayscale):
        self.task_id = task_id
        self.player_id = player_id
        self.num_players = num_players
        self.make_env_func = make_env_func
        self.seed = seed
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.grayscale = grayscale

        self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
        self.process = Process(target=self.start, daemon=True)
        self.process.start()

    def _init(self, port):
        log.info('Initializing env for player %d...', self.player_id)
        env = self.make_env_func(level=self.task_id, player_id=self.player_id, port=port,
                                 num_players=self.num_players, skip_frames=self.skip_frames,
                                 stack_frames=self.stack_frames, grayscale=self.grayscale)
        env.seed(self.seed + self.player_id)
        return env

    def _terminate(self, env):
        log.info('Stop env for player %d...', self.player_id)
        env.close()
        log.info('Env with player %d terminated!', self.player_id)

    @staticmethod
    def _get_info(env):
        """Specific to custom VizDoom environments."""
        info = {}
        if hasattr(env.unwrapped, 'get_info_all'):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def start(self):
        env = None

        while True:
            arg, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init(arg)  # arg = port
                self.task_queue.task_done()
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate(env)
                self.task_queue.task_done()
                break

            if task_type == TaskType.RESET:
                results = env.reset(mode=arg)
            elif task_type == TaskType.INFO:
                results = self._get_info(env)
            elif task_type == TaskType.SUB_STEP or task_type == TaskType.STEP:
                # collect obs, reward, done, and info
                env.unwrapped.full_step = task_type == TaskType.STEP
                results = env.step(arg)  # arg = action
            elif task_type == TaskType.RENDER:
                img = env.unwrapped.game.get_state().screen_buffer.transpose(1,2,0)

                results = img
            elif task_type == TaskType.GET_OBS:
                obs = env.get_obs()
                results = obs
            else:
                raise Exception(f'Unknown task type {task_type}')

            self.result_queue.put(results)
            self.task_queue.task_done()


class VizdoomMultiAgentEnv:
    def __init__(self, task_id=0, env_id=0, num_players=2, seed=0,
                 make_env_func=make_component_env, lock=None, skip_frames=4,
                 stack_frames=False, grayscale=False):
        self.task_id = task_id
        self.env_id = env_id  # distinguish from multiple concurrent envs (used to set port)
        self.num_players = num_players  # TODO: fix hardcoded numbers that assume 2 players
        self.seed = seed
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.grayscale = grayscale
        self.viewer = None
        if lock is None:
            lock = multiprocessing.Lock()
        self.lock = lock

        env = make_env_func(level=task_id, player_id=-1, num_players=num_players, skip_frames=skip_frames,
                            stack_frames=stack_frames, grayscale=grayscale)  # temporary
        self.action_space = env.action_space
        self.observation_space = spaces.Tuple((env.observation_space,
                                               spaces.Box(0, 1, (self.num_players * 2,),
                                                          dtype=np.float32)))
        self.state_space = spaces.Box(0, 1, (self.num_players * (10 + env.unwrapped.x_bins + env.unwrapped.y_bins),), dtype=np.float32)
        env.close()

        self.workers = [MultiAgentEnvWorker(task_id, i, num_players, make_env_func, seed, skip_frames,
                                            stack_frames, grayscale) for i in range(num_players)]

        with self.lock:
            port = 5029 + self.env_id
            # check if port is in use
            # we possess lock while initializing so no other process should be searching for a port concurrently
            while (port in [c.laddr.port for c in psutil.net_connections()]):
                port += 1
            for worker in self.workers:
                worker.task_queue.put((port, TaskType.INIT))
                sleep(0.1)  # just in case
            for worker in self.workers:
                worker.task_queue.join()


        log.info('%d agent workers initialized!', len(self.workers))

    def await_tasks(self, data, task_type, timeout=None):
        """
        Task result is always a tuple of lists, e.g.:
        (
            [0th_agent_obs, 1st_agent_obs, ... ,]
            [0th_agent_reward, 1st_agent_reward, ... ,]
            ...
        )

        If your "task" returns only one result per agent (e.g. reset() returns only the observation),
        the result will be a tuple of lenght 1. It is a responsibility of the caller to index appropriately.

        """
        if data is None:
            data = [None for i in range(self.num_players)]

        assert len(data) == self.num_players

        for i, worker in enumerate(self.workers[1:], start=1):
            worker.task_queue.put((data[i], task_type))
        self.workers[0].task_queue.put((data[0], task_type))

        result_lists = None
        for i, worker in enumerate(self.workers):
            worker.task_queue.join()
            results = safe_get(
                worker.result_queue,
                timeout=0.02 if timeout is None else timeout,
                msg=f'Takes a surprisingly long time to process task {task_type}, retry...',
            )

            worker.result_queue.task_done()

            if not isinstance(results, (tuple, list)):
                results = [results]

            if result_lists is None:
                result_lists = tuple([] for _ in results)

            for j, r in enumerate(results):
                result_lists[j].append(r)

        return result_lists

    def info(self):
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    def reset(self, mode='algo'):
        img_obs = self.await_tasks([mode for _ in self.workers], TaskType.RESET)[0]
        return self.get_st_obs(img_obs=img_obs)

    def get_st_obs(self, img_obs=None, infos=None):
        if infos is None:
            infos = self.await_tasks(None, TaskType.INFO)[0]
        if img_obs is None:
            img_obs = self.await_tasks(None, TaskType.GET_OBS)[0]
        vect_obs = [np.concatenate((inf['item1'], inf['item2'])) for inf in infos]
        obs = list(zip(img_obs, vect_obs))

        state = np.concatenate([np.concatenate((inf['pos'], inf['vel'],
                                                inf['orient'], inf['item1'],
                                                inf['item2'],
                                                inf['onehot_pos']))
                                for inf in infos])
        return state, obs

    def step(self, actions):
        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.SUB_STEP)  # still accumulates intermediate frames to frame stack
        img_obs, rews, dones, infos = self.await_tasks(actions, TaskType.STEP)
        done = all(dones)  # should all be the same in coop envs
        rew = np.mean(rews)  # should all be the same in coop envs
        self.visit_counts = np.stack([info['visit_counts'] for info in infos], axis=0)
        state, obs = self.get_st_obs(img_obs=img_obs, infos=infos)
        info = {}
        info['visit_count_lookup'] = [inf['hist_inds'] for inf in infos]
        info['n_found_treasures'] = [inf['n_found_treasures'] for inf in infos]
        return state, obs, rew, done, info

    def render(self, mode='human'):
        imgs = self.await_tasks(None, TaskType.RENDER)[0] # each agent's view
        img = np.concatenate(imgs, axis=1)

        h, w = img.shape[:2]
        render_w = 1600

        if w < render_w:
            render_h = int(1600 * h / w)
            img = cv2.resize(img, (render_w, render_h))
        if mode == 'rgb_array':
            return img

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer(maxwidth=1600)
        self.viewer.imshow(img)

    def close(self):
        log.info('Stopping multi env...')

        with self.lock:
            for worker in self.workers:
                worker.task_queue.put((None, TaskType.TERMINATE))
                sleep(0.1)
            for worker in self.workers:
                worker.process.join()

        if self.viewer is not None:
            self.viewer.close()

