import numpy as np
import gym
import sys
import os
import copy
import itertools
import string
import pygame
from collections import deque, OrderedDict
from io import StringIO
from gym import Env, spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

B = 10000000
SPAN = 1
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
GET = 4
ACTION_NAMES= ['UP', 'DOWN', 'LEFT', 'RIGHT', 'GET']

class ImgSprite(pygame.sprite.Sprite):
    def __init__(self, rect_pos=(5, 5, 64, 64)):
        super(ImgSprite, self).__init__()
        self.image = None
        self.rect = pygame.Rect(*rect_pos)

    def update(self, image):
        if isinstance(image, str):
            self.image = load_pygame_image(image)
        else:
            self.image = pygame.surfarray.make_surface(image)

class Render(object):
    def __init__(self, size=(320, 320)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.group = pygame.sprite.Group(ImgSprite()) # the group of all sprites

    def render(self, img):
        img = np.asarray(img).transpose(1, 0, 2)
        self.group.update(img)
        self.group.draw(self.screen)
        pygame.display.flip()
        e = pygame.event.poll()

def chunk(l, n):
    sz = len(l)
    assert sz % n == 0, 'cannot be evenly chunked'
    for i in range(0, sz, n):
        yield l[i:i+n]

class Agent():
    def __init__(self, agent_id, targets):
        self.agent_id = agent_id
        self.targets = targets
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None
        self.last_action = None
        self.done = False
        self.start_pos = None
        self.found_treasures = []

class Pit():
    def __init__(self, mag=0.05):
        self.prob_open = 0.0
        self.delta_mean = mag
        self.delta_std = mag
        self.is_open = False

    def tick(self):
        self.is_open = np.random.uniform() < self.prob_open
        if self.is_open:
            self.prob_open = 0.0
        else:
            self.prob_open += np.random.normal(self.delta_mean, self.delta_std)
            self.prob_open = np.clip(self.prob_open, 0.0, 1.0)



def color_interpolate(x, start_color, end_color):
    assert ( x <= 1 ) and ( x >= 0 )
    if not isinstance(start_color, np.ndarray):
        start_color = np.asarray(start_color[:3])
    if not isinstance(end_color, np.ndarray):
        end_color = np.asarray(end_color[:3])
    return np.round( (x * end_color + (1 - x) * start_color) * 255.0 ).astype(np.uint8)

CUR_DIR = os.path.dirname(__file__)
ACT_DICT = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
render_map_funcs = {
              '%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.5, .5, .5])),
              ' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
              '#': lambda x: color_interpolate(x, np.array([73, 49, 28]) / 255.0, np.array([219, 147, 86]) / 255.0),
              #'%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.3, .3, .3])),
              #' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.02)),
              #'#': lambda x: color_interpolate(x, np.array([219, 147, 86]) / 255.0, np.array([219, 147, 86]) / 255.0),
              # 'A': lambda x: (np.asarray(plt.cm.Reds(0.8)[:3]) * 255 ).astype(np.uint8),
              # 'B': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              # 'C': lambda x: (np.asarray(plt.cm.Greens(0.8)[:3]) * 255 ).astype(np.uint8),
              # 'D': lambda x: (np.asarray(plt.cm.Wistia(0.8)[:3]) * 255 ).astype(np.uint8),
              '1': lambda x: (np.asarray(plt.cm.Reds(0.8)[:3]) * 255 ).astype(np.uint8),
              '2': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              '3': lambda x: (np.asarray(plt.cm.Greens(0.8)[:3]) * 255 ).astype(np.uint8),
              '4': lambda x: (np.asarray(plt.cm.Oranges(0.8)[:3]) * 255 ).astype(np.uint8),
              'A': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255 ).astype(np.uint8),
              'B': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255 ).astype(np.uint8),
              'C': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255 ).astype(np.uint8),
              'D': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255 ).astype(np.uint8),
              # '1': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              # '2': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              # '3': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              # '4': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
             }# hard-coded

render_map = { k: render_map_funcs[k](1) for k in render_map_funcs }

def construct_render_map(vc):
    np_random = np.random.RandomState(9487)
    pertbs = dict()
    for i in range(20):
        pertb = dict()
        for c in render_map:
            if vc:
                pertb[c] = render_map_funcs[c](np_random.uniform(0, 1))
            else:
                pertb[c] = render_map_funcs[c](0)
        pertbs[i] = pertb
    return pertbs

# TODO: need_get, hindsight

def read_map(filename):
    m = []
    with open(filename) as f:
        for row in f:
            m.append(list(row.rstrip()))

    return m

def dis(a, b, p=2):
    res = 0
    for i, j in zip(a, b):
        res += np.power(np.abs(i-j), p)
    return np.power(res, 1.0/p)

def not_corner(m, i, j):
    if i == 0 or i == len(m)-1 or j == 0 or j == len(m[0])-1:
        return False
    if m[i-1][j] == '#' or m[i+1][j] == '#' or m[i][j-1] == '#' or m[i][j+1] == '#':
        return False
    return True

def build_gaussian_grid(grid, mean, std_coeff):
    row, col = grid.shape
    x, y = np.meshgrid(np.arange(row), np.arange(col))
    d = np.sqrt(x*x + y*y)
    return np.exp(-((x-mean[0]) ** 2 + (y-mean[1]) ** 2)/(2.0 * (std_coeff * min(row, col)) ** 2))

# roll list when task length is 2
def roll_list(l, n):
    res = []
    l = list(chunk(l, n-1))
    for i in range(n-1):
        for j in range(n):
            res.append(l[j][(i+j)%(n-1)])
    return res

class GridWorld(Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(
            self,
            map_inds,
            task_length=2,
            train_combos=None,
            test_combos=None,
            window=1,
            gaussian_img=True,
            reward_config=None,
            need_get=True,
            stay_act=False,
            joint_count=False,
            seed=0,
            task_config=1,
            num_agents=2,
            rand_trans=0.1,  # probability of random transition
            ):
        self.seed(seed)
        self.task_config = task_config
        self.num_agents = num_agents
        self.dist_mtx = np.zeros((num_agents, num_agents))
        if task_config == 1:
            num_obj_types = num_agents
            agent_targets = [list(range(num_obj_types)) for i in range(num_agents)]
        elif task_config == 2:
            num_obj_types = num_agents
            agent_targets = [list(range(num_obj_types)) for i in range(num_agents)]
        elif task_config == 3:
            num_obj_types = num_agents
            agent_targets = list(range(num_agents))

        self.agents = [Agent(agent_id=i + 1, targets=agent_targets[i]) for i in range(num_agents)]
        self.map_names = ['map%i_%i_multi' % (i, num_obj_types) for i in map_inds]
        self.maps = [read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(m))) for m in self.map_names]
        self.num_obj_types = num_obj_types
        self.task_length = task_length
        assert task_length <= num_obj_types, 'task length ({}) should be shorter than number of object types ({})'.format(task_length, num_obj_types)
        self.tasks = list(itertools.permutations(list(range(num_obj_types)), task_length))
        self.task_desc = list(itertools.permutations(list(string.ascii_uppercase[:num_obj_types]), task_length))
        if task_length == 2: # hardcoded preprocess
            self.tasks = roll_list(self.tasks, num_obj_types)
            self.task_desc = roll_list(self.task_desc, num_obj_types)
        self.train_combos = train_combos
        self.test_combos = test_combos
        self.n_train_combos = len(train_combos)
        self.n_test_combos = len(test_combos)
        self.img_stack = deque(maxlen=window)
        self.window = window
        self.gaussian_img = gaussian_img
        if reward_config is None:
            reward_config = {'wall_penalty': 0.0, 'time_penalty': -0.1, 'complete_sub_task': 2., 'get_same_treasure': 5., 'get_treasure': 10., 'fail': -10.}
        self.reward_config = reward_config
        self.need_get = need_get  # need to explicitly act to pick up treasure
        self.stay_act = stay_act  # separate action for staying put
        self.joint_count = joint_count
        self.rand_trans = rand_trans
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_obj_types,), dtype=np.float32)
        self.action_space = spaces.Discrete(5) if (need_get or stay_act) else spaces.Discrete(4)
        # scene, task
        self.row, self.col = len(self.maps[0]), len(self.maps[0][0]) # make sure all the maps you load are of the same size
        self.m = None
        self.task = None
        self.map_id = None
        self.task_id = None
        self._render = None
        self.found_treasures = []
        self.time = 0
        # keep track of how many times each agent has visited each cell
        if self.joint_count:
            self.visit_counts = np.zeros(num_agents * [self.row, self.col])
        else:
            self.visit_counts = np.zeros((num_agents, self.row, self.col))
        self.reset()

    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return seed

    def sample(self, train=True):
        if train:
            index = self.train_combos[self.random.randint(self.n_train_combos)]
        else:
            index = self.test_combos[self.random.randint(self.n_test_combos)]
        self.set_index(index)

    def set_index(self, index):
        self.map_id, self.task_id = index
        self.m = copy.deepcopy(self.maps[self.map_id])
        self.task = np.asarray(copy.deepcopy(self.tasks[self.task_id]))

    def _set_up_map(self, sample_pos):

        self.mask = np.ones(self.num_obj_types, dtype=np.uint8)
        self.wall = np.zeros((self.row, self.col))
        self.pos_candidates = [] # for object and task
        self.pits = OrderedDict()

        self.pos = []
        for i in range(self.num_obj_types):
            self.pos.append(())

        for i in range(len(self.m)):
            for j in range(len(self.m[i])):
                if self.m[i][j].isnumeric():
                    ai = int(self.m[i][j]) - 1
                    self.agents[ai].x = i
                    self.agents[ai].y = j
                    self.agents[ai].start_pos = (i, j)
                    self.m[i][j] = ' '
                elif self.m[i][j] == '#':
                    self.wall[i][j] = 1
                elif self.m[i][j] == '!':
                    rand_walk_mag = 0.05 if self.task_config == 1 else 0.005
                    self.pits[(i, j)] = Pit(mag=rand_walk_mag)
                elif self.m[i][j].isalpha():
                    pos_idx = ord(self.m[i][j])-ord('A')
                    self.pos[pos_idx] = (i,j)
                elif self.m[i][j] == ' ' or self.m[i][j].isalpha(): #and not_corner(self.m, i, j):
                    self.pos_candidates.append((i, j))

        if sample_pos:
            for agent in self.agents:
                agent.x, agent.y = self.pos_candidates[self.random.randint(len(self.pos_candidates))]

        self.up = []
        self.down = []
        self.left = []
        self.right = []
        for s in self.m: # distance
            self.up.append(B * np.ones(len(s)))
            self.down.append(B * np.ones(len(s)))
            self.left.append(B * np.ones(len(s)))
            self.right.append(B * np.ones(len(s)))

        for i in range(len(self.m)):
            for j in range(len(self.m[i])):
                if self.m[i][j] == '#':
                    self.up[i][j] = 0
                    self.left[i][j] = 0
                else:
                    if i > 0:
                        self.up[i][j] = self.up[i-1][j] + 1
                    if j > 0:
                        self.left[i][j] = self.left[i][j-1] + 1
        for i in reversed(range(len(self.m))):
            for j in reversed(range(len(self.m[i]))):
                if self.m[i][j] == '#':
                    self.down[i][j] = 0
                    self.right[i][j] = 0
                else:
                    if i < len(self.m) - 1:
                        self.down[i][j] = self.down[i+1][j] + 1
                    if j < len(self.m[i]) - 1:
                        self.right[i][j] = self.right[i][j+1] + 1


    def reset(self, index=None, sample_pos=False, train=True):
        if index is None:
            self.sample(train=train)
        else:
            self.set_index(index)
        self.found_treasures = []
        for a in self.agents:
            a.reset()

        self.time = 0

        self._set_up_map(sample_pos)

        if self.joint_count:
            visit_inds = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_inds] += 1
        else:
            for ia, agent in enumerate(self.agents):
                self.visit_counts[ia, agent.x, agent.y] += 1


        obs_list = [self.get_obs(a) for a in self.agents]

        return obs_list

    def get_obs(self, agent):
        out = np.zeros(self.num_obj_types)
        for t in agent.found_treasures:
            out[t] = 1
        out = np.concatenate([out, [agent.x / self.row, agent.y / self.col]])
        return out

    def process_get(self, agent):
        r = 0
        c = ord(self.m[agent.x][agent.y]) - ord('A')

        if self.task_config == 1:
            if c not in self.found_treasures:
                r += self.reward_config['get_treasure']
                self.found_treasures.append(c)
                agent.found_treasures.append(c)
                self.m[agent.x][agent.y] = ' '
        elif self.task_config == 2:
            if c not in agent.found_treasures:
                if len(self.found_treasures) == 0:
                    r += self.reward_config['get_treasure']
                    self.found_treasures.append(c)
                    agent.found_treasures.append(c)
                elif c in self.found_treasures:
                    r += self.reward_config['get_treasure']
                    agent.found_treasures.append(c)
        elif self.task_config == 3:
            if c == agent.targets:
                r += self.reward_config['get_treasure']
                self.found_treasures.append(c)
                agent.found_treasures.append(c)
                self.m[agent.x][agent.y] = ' '

        return c, r

    def step(self, action_list):
        obs_list = []
        if not all(type(a) is int for a in action_list):
            # get integer action (rather than onehot)
            action_list = [a.argmax() for a in action_list]
        if self.rand_trans > 0.0:
            action_list = [a if np.random.uniform() > self.rand_trans else
                           np.random.randint(self.action_space.n)
                           for a in action_list]

        total_reward = 0
        # try initial moving actions
        for idx, agent in enumerate(self.agents):
            total_reward += self.reward_config['time_penalty']
            action = action_list[idx]
            self.last_action = ACTION_NAMES[action]

            agent.prev_x = agent.x
            agent.prev_y = agent.y

            if action == 0:
                if agent.x > 0:
                    if self.m[agent.x-1][agent.y] != '#':
                        agent.x -= 1
                    else:
                        total_reward += self.reward_config['wall_penalty']            
            elif action == 1:
                if agent.x < len(self.m)-1:
                    if self.m[agent.x+1][agent.y] != '#':
                        agent.x += 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
            elif action == 2:
                if agent.y > 0:
                    if self.m[agent.x][agent.y-1] != '#':
                        agent.y -= 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
            elif action == 3:
                if agent.y < len(self.m[agent.x])-1:
                    if self.m[agent.x][agent.y+1] != '#':
                        agent.y += 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
        for pit in self.pits.values():
            pit.tick()
        # check for pits
        for ia, agent in enumerate(self.agents):
            if self.m[agent.x][agent.y] == '!':
                curr_pit = self.pits[(agent.x, agent.y)]
                if curr_pit.is_open:
                    agent.x, agent.y = agent.start_pos

        # check for collisions
        for ia, agent in enumerate(self.agents):
            for oa in range(ia + 1, self.num_agents):
                other = self.agents[oa]
                dist = abs(agent.x - other.x) + abs(agent.y - other.y)
                if dist == 0:
                    # move non-incumbent agents back to their previous positions
                    if (agent.x, agent.y) == (agent.prev_x, agent.prev_y):
                        (other.x, other.y) = (other.prev_x, other.prev_y)
                    elif (other.x, other.y) == (other.prev_x, other.prev_y):
                        (agent.x, agent.y) = (agent.prev_x, agent.prev_y)
                    else: # if neither agent is incumbent, then randomly choose one to take the disputed position
                        chosen_agent = self.random.choice([agent, other])
                        (chosen_agent.x, chosen_agent.y) = (chosen_agent.prev_x, chosen_agent.prev_y)
        # calculate distances and add to visit counts
        for ia, agent in enumerate(self.agents):
            self.dist_mtx[ia][ia] = 0
            for oa in range(ia + 1, self.num_agents):
                other = self.agents[oa]
                dist = abs(agent.x - other.x) + abs(agent.y - other.y)
                self.dist_mtx[ia][oa] = dist
                self.dist_mtx[oa][ia] = dist

            # get observations
            obs = self.get_obs(agent)
            obs_list.append(obs)

            action = action_list[idx]
            # process 'get treasure' action, and assign treasure-specific rewards
            if (action == 4 or not self.need_get) and self.m[agent.x][agent.y].isalpha():
                c, r = self.process_get(agent) # not adding to previous r
                total_reward += r

        if self.task_config == 1:
            done = len(self.found_treasures) == self.num_obj_types
        elif self.task_config == 2:
            done = (len(self.found_treasures) == 1) and all(self.found_treasures[0] in a.found_treasures for a in self.agents)
        elif self.task_config == 3:
            done = all(len(a.found_treasures) == 1 for a in self.agents)

        if self.joint_count:
            visit_inds = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_inds] += 1
        else:
            for ia, agent in enumerate(self.agents):
                self.visit_counts[ia, agent.x, agent.y] += 1
        infos = {}
        infos['visit_count_lookup'] = [[a.x, a.y] for a in self.agents]
        infos['n_found_treasures'] = [len(a.found_treasures) for a in self.agents]
        self.time += 1
        return (obs_list, total_reward, done, infos)

    def render(self, mode='human', close=False, verbose=True):

        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for idx, agent in enumerate(self.agents):
            if verbose:
                out = 'scene: {}, task: {}, index: {}\n'.format(self.map_id, self.task_desc[self.task_id], self.index())
                out += 'last action: {}\n'.format(agent.last_action) if agent.last_action is not None else ''
                obs = self.get_obs(agent)
                out += 'pos: ({}, {})\n'.format(obs[0], obs[1])
                out += 'wall distance: {}\n'.format(obs[2:6])
                out += 'distance to all goals: {}\n'.format(obs[6:6+self.num_obj_types])
            else:
                out = ''
            for x in range(len(self.m)):
                for y in range(len(self.m[x])):
                    if x == agent.x and y == agent.y:
                        out += '%'
                    else:
                        out += self.m[x][y] 
                out += "\n"
            outfile.write(out)
        if mode != 'human':
            return outfile

    def init_render(self, block_size=16):
      if self._render is None:
          self._render = Render(size=(self.row * block_size, self.col * block_size))
      return self

    # color table: https://www.rapidtables.com/web/color/RGB_Color.html
    def pretty_render(self, init_render=False, repeat=15):
        out = self.render(verbose=False, mode='ansi')
        world = np.zeros((self.row, self.col, 3))
        i, j = 0, 0
        for c in out.getvalue():
            if c == '\n':
                i += 1
                j = 0
            else:
                world[i, j, :] = render_map[c]
                j += 1
        world = world.repeat(repeat, 0).repeat(repeat, 1)
        if init_render:
            self.init_render()
            self._render.render(world)
        return world

    def render_map(self):
        out = self.render(verbose=False, mode='ansi')
        world = np.zeros((self.row, self.col, 3))
        i, j = 0, 0
        for c in out.getvalue():
            if c == '\n':
                i += 1
                j = 0
            else:
                if c not in ['#', ' ']: c = ' '
                world[i, j, :] = render_map[c]
                j += 1
        world = world.repeat(15, 0).repeat(15, 1)
        return world

    def index(self):
        return self.map_id, self.task_id

class EnvWrapper(gym.Wrapper):
    def pretty_render(self):
        return self.env.unwrapped.pretty_render()

    def index(self):
        return self.env.unwrapped.index()

    @property
    def map_names(self):
        return self.env.unwrapped.map_names

    @property
    def agents(self):
        return self.env.unwrapped.agents

    @property
    def num_agents(self):
        return len(self.env.unwrapped.agents)

    @property
    def tasks(self):
        return self.env.unwrapped.tasks

    @property
    def task_desc(self):
        return self.env.unwrapped.task_desc

    @property
    def window(self):
        return self.env.unwrapped.window


class VectObsEnv(EnvWrapper):
    def __init__(self, env, l=3, vc=False, block_size=16):
        super().__init__(env)
        self.row, self.col = self.env.unwrapped.row, self.env.unwrapped.col
        self.block_size = block_size
        window = self.env.unwrapped.window
        self.num_obj_types = self.env.unwrapped.num_obj_types
        self.obs_img_stacks = [deque(maxlen=window) for a in env.agents]
        self.state_space = spaces.Box(0, 1, ((4 + 5 + self.num_obj_types + self.row + self.col) * self.num_agents,),
                                      dtype=np.float32)
        # self.state_space = spaces.Box(0, 1, ((4 + 5 + self.num_obj_types + 2) * self.num_agents,),
        #                               dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, (4 + 5 + self.num_obj_types + 2 + (self.num_agents - 1) * 2,),
                                            dtype=np.float32)
        self.action_space = self.env.unwrapped.action_space
        self.l = l
        self.pertbs = construct_render_map(vc)
        self.base_img = None

    def _get_agent_state(self, agent):
        """
        Return info for an agent that is part of the global state and agent's observation
        """
        surr_walls = np.zeros(4)
        surr_pit_probs = np.zeros(5)
        env_map = self.env.unwrapped.m
        pits = self.env.unwrapped.pits
        x, y = agent.x, agent.y
        if env_map[x][y] == '!':
            surr_pit_probs[0] = pits[(x,y)].prob_open
        elif env_map[x][y+1] in ['#', '!']:
            if env_map[x][y+1] == '#':
                surr_walls[0] = 1
            else:
                surr_pit_probs[1] = pits[(x,y+1)].prob_open
        elif env_map[x+1][y] in ['#', '!']:
            if env_map[x+1][y] == '#':
                surr_walls[1] = 1
            else:
                surr_pit_probs[2] = pits[(x+1,y)].prob_open
        elif env_map[x][y-1] in ['#', '!']:
            if env_map[x][y-1] == '#':
                surr_walls[2] = 1
            else:
                surr_pit_probs[3] = pits[(x,y-1)].prob_open
        elif env_map[x-1][y] in ['#', '!']:
            if env_map[x-1][y] == '#':
                surr_walls[3] = 1
            else:
                surr_pit_probs[4] = pits[(x-1,y)].prob_open

        coll_treasure = np.zeros(self.num_obj_types)
        for t in agent.found_treasures:
            coll_treasure[t] = 1

        agent_state = np.concatenate([surr_walls, surr_pit_probs, coll_treasure])
        return agent_state

    def _get_state(self):
        agent_states = [self._get_agent_state(a) for a in self.env.agents]
        agent_coords = []
        for a in self.env.agents:
            agent_coords.append((np.arange(self.row) == a.x).astype(np.float32))
            agent_coords.append((np.arange(self.col) == a.y).astype(np.float32))
            # agent_coords.append(np.array([a.x / self.row, a.y / self.col] , dtype=np.float32))
        return np.concatenate(agent_states + agent_coords)

    def _get_agent_obs(self, agent):
        agent_state = self._get_agent_state(agent)

        agent_loc = np.array([agent.x / self.row, agent.y / self.col] , dtype=np.float32)
        other_locs = []
        for oa in self.env.agents:
            if oa is agent:
                continue
            rel_loc = np.array([oa.x - agent.x, oa.y - agent.y]) / self.l
            if np.abs(rel_loc).max() > 1.0:
                rel_loc = np.zeros(2)
            other_locs.append(rel_loc)
        out = np.concatenate([agent_state, agent_loc] + other_locs)
        return out

    def _get_obs(self):
        return [self._get_agent_obs(a) for a in self.env.agents]

    def _generate_base_img(self):
        """
        Generate base image of map to draw agents and objects onto
        """
        img = 255 * np.ones((3, self.row, self.col), dtype=np.uint8)
        pertb = self.pertbs[self.env.unwrapped.map_id]
        m = self.env.unwrapped.m

        for x in range(len(self.env.unwrapped.m)):
            for y in range(len(self.env.unwrapped.m[x])):
                if not m[x][y].isalnum() and not m[x][y] == '!':  # not an agent's starting position, treasure, or pit
                    img[:, x, y] = pertb[m[x][y]]
        return img

    def reset(self, *args, **kwargs):
        o = self.env.reset(*args, **kwargs)
        return self._get_state(), self._get_obs()

    def get_st_obs(self):
        return self._get_state(), self._get_obs()

    def step(self, actions):
        next_o, r, done, info = self.env.step(actions)
        return self._get_state(), self._get_obs(), r, done, info

    def render(self):
        self.env.unwrapped.init_render(block_size=self.block_size)
        if self.base_img is None:
            self.base_img = self._generate_base_img()
        img = self.base_img.copy()
        pertb = self.pertbs[self.env.unwrapped.map_id]
        m = self.env.unwrapped.m

        for x, y in self.env.unwrapped.pos:  # redraw treasure in case it has been picked up
            img[:, x, y] = pertb[m[x][y]]

        for x,y in self.env.unwrapped.pits:
            pit = self.env.unwrapped.pits[(x,y)]
            if not pit.is_open:
                img[:, x, y] = int((1 - pit.prob_open) * 200) + 55

        for agent in self.agents:
            x, y = agent.x, agent.y
            img[:, x, y] = pertb[str(agent.agent_id)]

        for x,y in self.env.unwrapped.pits:
            pit = self.env.unwrapped.pits[(x,y)]
            if pit.is_open:
                img[:, x, y] = 0
        self.render_img = img.transpose(1,2,0).repeat(self.block_size, 0).repeat(self.block_size, 1)

        self.env.unwrapped._render.render(self.render_img)
