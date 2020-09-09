import numpy as np
import gym
import os
import copy
import pygame
from collections import deque, OrderedDict
from itertools import product
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
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None
        self.last_action = None
        self.done = False
        self.start_pos = None
        self.curr_tier = 0
        self.found_treasures = []

class Pit():
    def __init__(self, mag=0.05):
        self.prob_open = 0.0
        self.set_mag(mag)
        self.is_open = False

    def set_mag(self, mag):
        self.delta_mean = mag
        self.delta_std = mag

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

render_map_funcs = {
              '$': lambda x: color_interpolate(x, np.array([0.7, 0.35, 0.0]), np.array([0.7, 0.35, 0.0])),
              '%': lambda x: color_interpolate(x, np.array([0.7, 0.35, 0.0]), np.array([0.7, 0.35, 0.0])),
              '&': lambda x: color_interpolate(x, np.array([0.7, 0.35, 0.0]), np.array([0.7, 0.35, 0.0])),
              "'": lambda x: color_interpolate(x, np.array([0.7, 0.35, 0.0]), np.array([0.7, 0.35, 0.0])),
              ' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
              '#': lambda x: color_interpolate(x, np.array([73, 49, 28]) / 255.0, np.array([219, 147, 86]) / 255.0),
              #'%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.3, .3, .3])),
              #' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.02)),
              #'#': lambda x: color_interpolate(x, np.array([219, 147, 86]) / 255.0, np.array([219, 147, 86]) / 255.0),
              # 'A': lambda x: (np.asarray(plt.cm.Reds(0.8)[:3]) * 255).astype(np.uint8),
              # 'B': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
              # 'C': lambda x: (np.asarray(plt.cm.Greens(0.8)[:3]) * 255).astype(np.uint8),
              # 'D': lambda x: (np.asarray(plt.cm.Wistia(0.8)[:3]) * 255).astype(np.uint8),
              '1': lambda x: (np.asarray(plt.cm.Reds(0.8)[:3]) * 255).astype(np.uint8),
              '2': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
              '3': lambda x: (np.asarray(plt.cm.Greens(0.8)[:3]) * 255).astype(np.uint8),
              '4': lambda x: (np.asarray(plt.cm.Wistia(0.8)[:3]) * 255).astype(np.uint8),
              'A': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'B': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'C': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'D': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'E': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'F': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'G': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              'H': lambda x: (np.asarray(plt.cm.Wistia(0.2)[:3]) * 255).astype(np.uint8),
              # '1': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
              # '2': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
              # '3': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
              # '4': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255).astype(np.uint8),
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


def get_neighboring_points(base_map, x, y, d=1):
    points = []
    if x - d >= 0:
        points.append((x - d, y))
    if y - d >= 0:
        points.append((x, y - d))
    if x + d < base_map.shape[0]:
        points.append((x + d, y))
    if y + d < base_map.shape[1]:
        points.append((x, y + d))
    return points


def can_make_path(base_map, curr_pos, next_pos):
    if next_pos[0] in (0, base_map.shape[0] - 1) or next_pos[1] in (0, base_map.shape[1] - 1):
        # can't leave outer walls
        return False
    cx, cy = curr_pos
    nx, ny = next_pos
    nps = get_neighboring_points(base_map, nx, ny)
    if (cx, cy) in nps:
        nps.pop(nps.index((cx, cy)))
    return base_map[nx, ny] == '#' and all(base_map[npx, npy] == '#' for npx, npy in nps)


def generate_rand_map(size=(20, 20), num_agents=2, num_tiers=2, paths_per_tier=2, pit_prob=0.15, rs=None):
    if rs is None:
        rs = np.random.RandomState(0)

    num_paths = paths_per_tier * num_tiers
    assert num_paths <= 8

    base_map = np.array([['#' for _ in range(size[1])] for _ in range(size[0])])
    path_map = np.array([['#' for _ in range(size[1])] for _ in range(size[0])])
    dist_map = -np.ones(size)

    start_box_size = 1
    while start_box_size**2 <= num_agents or (start_box_size * 4) // 2 <= num_paths:
        start_box_size += 1

    box_corner = ((size[0] - start_box_size) // 2, (size[1] - start_box_size) // 2)
    start_cands = list(product(range(box_corner[0], box_corner[0] + start_box_size),
                               range(box_corner[1], box_corner[1] + start_box_size)))
    non_path_cands = list(product(range(box_corner[0] + 1, box_corner[0] + start_box_size - 1),
                                  range(box_corner[1] + 1, box_corner[1] + start_box_size - 1)))
    path_cands = start_cands.copy()
    for npc in non_path_cands:
        path_cands.remove(npc)
    for pc in path_cands:
        dist_map[pc[0], pc[1]] = 0

    for x, y in start_cands:
        base_map[x][y] = ' '
        path_map[x][y] = ' '

    for i in range(1, num_agents + 1):
        start_loc = start_cands.pop(rs.randint(len(start_cands)))
        base_map[start_loc[0], start_loc[1]] = str(i)

    path_bases = sum([[(p, path_start) for p in get_neighboring_points(base_map, *path_start) if can_make_path(base_map, path_start, p)] for path_start in path_cands], [])
    path_stacks = [path_bases.copy() for _ in range(num_paths)]
    for stack in path_stacks:
        rs.shuffle(stack)

    while any(len(stack) > 0 for stack in path_stacks):
        for i, stack in enumerate(path_stacks):
            while len(stack) > 0:
                pc, prev = stack.pop(-1)
                # check if path still possible here
                if can_make_path(base_map, prev, pc):
                    path_map[pc[0], pc[1]] = str(i + 1)
                    if rs.uniform() < pit_prob:
                        base_map[pc[0], pc[1]] = '!'
                    else:
                        base_map[pc[0], pc[1]] = ' '
                    path_cands = [p for p in get_neighboring_points(base_map, *pc) if can_make_path(base_map, pc, p)]
                    rs.shuffle(path_cands)
                    curr_len = dist_map[prev[0], prev[1]]
                    dist_map[pc[0], pc[1]] = curr_len + 1
                    if curr_len == 0:
                        # don't start other new paths from center
                        stack.clear()
                        if i >= paths_per_tier:
                            tier_num = (i // paths_per_tier) - 1
                            base_map[pc[0], pc[1]] = chr(ord('$') + tier_num)
                    stack += [(npc, pc) for npc in path_cands]
                    break

    obj_locs = []
    for path_num in range(1, num_paths + 1):
        max_dist = dist_map[path_map == str(path_num)].max()
        obj_loc = list(zip(*np.where(np.logical_and(dist_map == max_dist, path_map == str(path_num)))))[0]
        obj_locs.append(obj_loc)
        base_map[obj_loc[0], obj_loc[1]] = chr(ord('A') + (path_num - 1))
    return base_map, dist_map, path_map


class GridWorld(Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(
            self,
            map_ind,
            window=1,
            reward_config=None,
            need_get=True,
            stay_act=False,
            joint_count=False,
            seed=0,
            task_config=1,
            num_agents=2,
            rand_trans=0.1,  # probability of random transition
            size=(20, 20),  # size of map if generating
            ):
        self.seed(seed)
        self.task_config = task_config
        self.num_agents = num_agents
        self.dist_mtx = np.zeros((num_agents, num_agents))
        if task_config == 1:
            self.task_tiers = [1]
            self.intermediate_rews = True
            self.num_obj_types = num_agents
            self.obj_tiers = [list(range(num_agents))]
        elif task_config == 2:
            self.task_tiers = [2]
            self.intermediate_rews = True
            self.num_obj_types = num_agents
            self.obj_tiers = [list(range(num_agents))]
        elif task_config == 3:
            self.task_tiers = [3]
            self.intermediate_rews = True
            self.num_obj_types = num_agents
            self.obj_tiers = [list(range(num_agents))]
        elif task_config == 4:
            assert map_ind == -1, "This task only supports randomly generated maps (set map_ind = -1)"
            num_paths = 8
            assert num_paths % num_agents == 0
            num_tiers = num_paths // num_agents
            poss_tasks = [1, 2]
            self.random.shuffle(poss_tasks)
            # ensures that task type switches at each tier
            self.task_tiers = [poss_tasks[0] if i % 2 == 0 else poss_tasks[1] for i in range(num_tiers)]
            self.intermediate_rews = True
            self.num_obj_types = num_paths
            self.obj_tiers = [list(range(j * num_agents, (j + 1) * num_agents)) for j in range(len(self.task_tiers))]
        self.num_tiers = len(self.task_tiers)

        self.agents = [Agent(agent_id=i + 1) for i in range(num_agents)]
        if map_ind == -1:
            self.map_name = 'rand'
            path_lens = [0, float('inf')]
            num_paths = num_agents * self.num_tiers
            while min(path_lens) < 0.8 * max(path_lens):
                base_map, dist_map, path_map = generate_rand_map(
                    size=size, num_agents=num_agents,
                    paths_per_tier=num_agents, num_tiers=self.num_tiers,
                    rs=self.random)
                path_lens = [dist_map[path_map == str(i)].max() for i in range(1, (num_paths) + 1)]
            self.map = base_map.tolist()
        else:
            self.map_name = 'map%i_%i_multi' % (map_ind, self.num_obj_types)
            self.map = read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(self.map_name)))

        self.door_types = [chr(ord('$') + i) for i in range(12)]
        self.img_stack = deque(maxlen=window)
        self.window = window
        if reward_config is None:
            reward_config = {'wall_penalty': 0.0, 'time_penalty': -0.1, 'complete_sub_task': 10., 'get_same_treasure': 5., 'get_treasure': 10., 'fail': -10.}
        self.reward_config = reward_config
        self.need_get = need_get  # need to explicitly act to pick up treasure
        self.stay_act = stay_act  # separate action for staying put
        self.joint_count = joint_count
        self.rand_trans = rand_trans
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_obj_types,), dtype=np.float32)
        self.action_space = spaces.Discrete(5) if (need_get or stay_act) else spaces.Discrete(4)
        # scene, task
        self.row, self.col = len(self.map), len(self.map[0])
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

    def _get_pit_mag(self):
        return 0.05 if self.curr_task == 1 else 0.005

    def _set_up_map(self, sample_pos):

        self.mask = np.ones(self.num_obj_types, dtype=np.uint8)
        self.wall = np.zeros((self.row, self.col))
        self.pos_candidates = []
        self.pits = OrderedDict()
        self.doors = [[] for _ in range(len(self.door_types))]

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
                    self.pits[(i, j)] = Pit(mag=self._get_pit_mag())
                elif self.m[i][j] in self.door_types:
                    self.doors[self.door_types.index(self.m[i][j])].append((i, j))
                elif self.m[i][j].isalpha():
                    pos_idx = ord(self.m[i][j]) - ord('A')
                    self.pos[pos_idx] = (i, j)
                elif self.m[i][j] == ' ' or self.m[i][j].isalpha(): #and not_corner(self.m, i, j):
                    self.pos_candidates.append((i, j))

        if sample_pos:
            for agent in self.agents:
                agent.x, agent.y = self.pos_candidates[self.random.randint(len(self.pos_candidates))]

    def reset(self, sample_pos=False):
        self.m = copy.deepcopy(self.map)
        self.found_treasures = []
        self.curr_tier = 0
        self.curr_task = self.task_tiers[self.curr_tier]
        self.curr_objs = self.obj_tiers[self.curr_tier]
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

        if self.curr_task == 1:
            if c not in self.found_treasures and c in self.curr_objs:
                r += self.reward_config['get_treasure']
                self.found_treasures.append(c)
                agent.found_treasures.append(c)
                self.m[agent.x][agent.y] = ' '
        elif self.curr_task == 2:
            if c not in agent.found_treasures and c in self.curr_objs:
                if all(ft not in self.curr_objs for ft in self.found_treasures):
                    r += self.reward_config['get_treasure']
                    self.found_treasures.append(c)
                    agent.found_treasures.append(c)
                elif c in self.found_treasures:
                    r += self.reward_config['get_treasure']
                    agent.found_treasures.append(c)
        elif self.curr_task == 3:
            if c == agent.agent_id - 1:
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
                    if self.m[agent.x - 1][agent.y] not in ['#'] + self.door_types:
                        agent.x -= 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
            elif action == 1:
                if agent.x < len(self.m) - 1:
                    if self.m[agent.x + 1][agent.y] not in ['#'] + self.door_types:
                        agent.x += 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
            elif action == 2:
                if agent.y > 0:
                    if self.m[agent.x][agent.y - 1] not in ['#'] + self.door_types:
                        agent.y -= 1
                    else:
                        total_reward += self.reward_config['wall_penalty']
            elif action == 3:
                if agent.y < len(self.m[agent.x]) - 1:
                    if self.m[agent.x][agent.y + 1] not in ['#'] + self.door_types:
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
                    else:  # if neither agent is incumbent, then randomly choose one to take the disputed position
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
                c, r = self.process_get(agent)  # not adding to previous r
                if self.intermediate_rews:
                    total_reward += r

        if self.curr_task == 1:
            task_done = all(obj in self.found_treasures for obj in self.curr_objs)
        elif self.curr_task == 2:
            curr_found = [obj for obj in self.curr_objs if obj in self.found_treasures]
            task_done = len(curr_found) == 1 and all(curr_found[0] in a.found_treasures for a in self.agents)
        elif self.curr_task == 3:
            agents_curr_found = [[obj for obj in self.curr_objs if obj in a.found_treasures] for a in self.agents]
            task_done = all(len(curr_found) == 1 for curr_found in agents_curr_found)

        if task_done:
            done = self.curr_tier == len(self.task_tiers) - 1
            if not self.intermediate_rews:
                total_reward += self.reward_config['complete_sub_task']
            self.curr_tier += 1
            if not done:
                door_locs = self.doors[self.curr_tier - 1]
                for x, y in door_locs:
                    self.m[x][y] = ' '
                self.curr_task = self.task_tiers[self.curr_tier]
                for pit in self.pits.values():
                    pit.set_mag(self._get_pit_mag())
                self.curr_objs = self.obj_tiers[self.curr_tier]
        else:
            done = False

        if self.joint_count:
            visit_inds = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_inds] += 1
        else:
            for ia, agent in enumerate(self.agents):
                self.visit_counts[ia, agent.x, agent.y] += 1
        infos = {}
        infos['visit_count_lookup'] = [[a.x, a.y] for a in self.agents]
        infos['n_found_treasures'] = [len(a.found_treasures) for a in self.agents]
        infos['tiers_completed'] = self.curr_tier
        self.time += 1
        return (obs_list, total_reward, done, infos)

    def init_render(self, block_size=16):
        if self._render is None:
            self._render = Render(size=(self.row * block_size, self.col * block_size))
        return self


class EnvWrapper(gym.Wrapper):
    @property
    def map_name(self):
        return self.env.unwrapped.map_name

    @property
    def agents(self):
        return self.env.unwrapped.agents

    @property
    def num_agents(self):
        return len(self.env.unwrapped.agents)

    @property
    def window(self):
        return self.env.unwrapped.window


class VectObsEnv(EnvWrapper):
    def __init__(self, env, l=3, vc=False, block_size=32):
        super().__init__(env)
        self.row, self.col = self.env.unwrapped.row, self.env.unwrapped.col
        self.block_size = block_size
        self.num_obj_types = self.env.unwrapped.num_obj_types
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
            surr_pit_probs[0] = pits[(x, y)].prob_open
        elif env_map[x][y + 1] in ['#', '!']:
            if env_map[x][y + 1] in ['#'] + self.env.unwrapped.door_types:
                surr_walls[0] = 1
            else:
                surr_pit_probs[1] = pits[(x, y + 1)].prob_open
        elif env_map[x + 1][y] in ['#', '!']:
            if env_map[x + 1][y] in ['#'] + self.env.unwrapped.door_types:
                surr_walls[1] = 1
            else:
                surr_pit_probs[2] = pits[(x + 1, y)].prob_open
        elif env_map[x][y - 1] in ['#', '!']:
            if env_map[x][y - 1] in ['#'] + self.env.unwrapped.door_types:
                surr_walls[2] = 1
            else:
                surr_pit_probs[3] = pits[(x, y - 1)].prob_open
        elif env_map[x - 1][y] in ['#', '!']:
            if env_map[x - 1][y] in ['#'] + self.env.unwrapped.door_types:
                surr_walls[3] = 1
            else:
                surr_pit_probs[4] = pits[(x - 1, y)].prob_open

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

        agent_loc = np.array([agent.x / self.row, agent.y / self.col], dtype=np.float32)
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
        pertb = self.pertbs[0]
        m = self.env.unwrapped.m

        for x in range(len(self.env.unwrapped.m)):
            for y in range(len(self.env.unwrapped.m[x])):
                if not m[x][y].isalnum() and not m[x][y] in ['!'] + self.env.unwrapped.door_types:  # not an agent's starting position, treasure, pit, or door
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

    def render(self, mode='human'):
        if mode == 'human':
            self.env.unwrapped.init_render(block_size=self.block_size)
        if self.base_img is None:
            self.base_img = self._generate_base_img()
        img = self.base_img.copy()
        pertb = self.pertbs[0]
        m = self.env.unwrapped.m

        for x, y in self.env.unwrapped.pos:
            img[:, x, y] = pertb[m[x][y]]

        for doorset in self.env.unwrapped.doors:
            for x, y in doorset:
                img[:, x, y] = pertb[m[x][y]]

        for x, y in self.env.unwrapped.pits:
            pit = self.env.unwrapped.pits[(x,y)]
            if not pit.is_open:
                img[:, x, y] = int((1. - pit.prob_open) * 200) + 55

        for agent in self.agents:
            x, y = agent.x, agent.y
            img[:, x, y] = pertb[str(agent.agent_id)]

        for x, y in self.env.unwrapped.pits:
            pit = self.env.unwrapped.pits[(x,y)]
            if pit.is_open:
                img[:, x, y] = 0

        self.render_img = img.transpose(1, 2, 0).repeat(self.block_size, 0).repeat(self.block_size, 1)

        if mode == 'human':
            self.env.unwrapped._render.render(self.render_img)
        elif mode == 'rgb_array':
            return self.render_img


if __name__ == '__main__':
    act_dict = {pygame.K_w: [0, 4, 4, 4],
                pygame.K_a: [2, 4, 4, 4],
                pygame.K_s: [1, 4, 4, 4],
                pygame.K_d: [3, 4, 4, 4],
                pygame.K_t: [4, 0, 4, 4],
                pygame.K_f: [4, 2, 4, 4],
                pygame.K_g: [4, 1, 4, 4],
                pygame.K_h: [4, 3, 4, 4],
                pygame.K_i: [4, 4, 0, 4],
                pygame.K_j: [4, 4, 2, 4],
                pygame.K_k: [4, 4, 1, 4],
                pygame.K_l: [4, 4, 3, 4],
                pygame.K_UP: [4, 4, 4, 0],
                pygame.K_LEFT: [4, 4, 4, 2],
                pygame.K_DOWN: [4, 4, 4, 1],
                pygame.K_RIGHT: [4, 4, 4, 3],
                }
    env = VectObsEnv(GridWorld(-1,
                               seed=0,
                               task_config=4,
                               num_agents=2,
                               rand_trans=0.0,
                               need_get=False,
                               stay_act=True), l=3)

    env.reset()
    img = env.render()

    done = False
    while not done:
        while True:
            # poll for actions (one agent at a time)
            events = pygame.event.get()
            actions = None
            for event in events:
                if event.type == pygame.KEYDOWN:
                    actions = act_dict.get(event.key, [4, 4, 4, 4])[:env.num_agents]
                    break
            if actions is not None:
                break
        _, _, rew, done, infos = env.step(actions)
        print(rew, infos['n_found_treasures'], infos['tiers_completed'])
        env.render()

    env.close()
