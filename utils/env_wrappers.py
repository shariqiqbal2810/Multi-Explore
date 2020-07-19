"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import psutil
import os
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            st, ob, reward, done, info = env.step(data)
            # if done:
            #     st, ob = env.reset()
            remote.send((st, ob, reward, done, info))
        elif cmd == 'reset':
            st, ob = env.reset()
            remote.send((st, ob))
        elif cmd == 'get_st_obs':
            st, ob = env.get_st_obs()
            remote.send((st, ob))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.state_space, env.observation_space, env.action_space))
        elif cmd == 'get_visit_counts':
            if hasattr(env, 'visit_counts'):
                remote.send(env.visit_counts)
            elif hasattr(env.unwrapped, 'visit_counts'):
                remote.send(env.unwrapped.visit_counts)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False # can't since vizdoom envs have their own daemonic subprocesses
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.state_space, observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def get_visit_counts(self):
        for remote in self.remotes:
            remote.send(('get_visit_counts', None))
        return sum(remote.recv() for remote in self.remotes)

    def step_async(self, actions, envs):
        for action, env_i in zip(actions, envs):
            self.remotes[env_i].send(('step', action))
        self.waiting = True

    def step_wait(self, envs):
        results = [self.remotes[i].recv() for i in envs]
        self.waiting = False
        state, obs, rews, dones, infos = zip(*results)
        # state and obs are tuples, so stack their components separately
        return (self._stack(state),
                self._stack(obs),
                np.array(rews), np.array(dones), infos)

    def _stack(self, items, *args):
        """
        Stack items received from multiple environments.
        items indexed as such: (n_envs, ..., numpy array with shape (*dims)) where '...' can be any number
        of arbitrary nested tuples/lists, to get a set of nested lists indexed as
        (..., numpy array w/ shape (n_envs, *dims))
        """
        if len(args) == 0:
            return self._stack(items, 0)
        sub_items = items
        for dim in args[::-1]:
            sub_items = sub_items[dim]
        if type(sub_items) in (tuple, list):
            return [self._stack(items, i, *args) for i in range(len(sub_items))]
        else:
            will_stack = []
            for i in range(len(items)):
                sub_items = items[i]
                for dim in args[:-1][::-1]:
                    sub_items = sub_items[dim]
                will_stack.append(sub_items)
            return np.stack(will_stack)

    def step(self, actions, env_mask=None):
        if env_mask is None:
            env_mask = np.ones(len(self.remotes))
        envs = np.where(env_mask)[0]
        self.step_async(actions, envs)
        return self.step_wait(envs)

    def reset(self, need_reset=None):
        if need_reset is None:
            need_reset = [True for _ in range(len(self.remotes))]
        for remote, nr in zip(self.remotes, need_reset):
            if nr:
                remote.send(('reset', None))
            else:
                remote.send(('get_st_obs', None))
        results = [remote.recv() for remote in self.remotes]
        state, obs = zip(*results)
        return (self._stack(state), self._stack(obs))

    def close(self, force=False):
        if self.closed:
            return
        if force:  # super ugly
            for p in self.ps:
                p.terminate()  # kill parallel environment workers
            # extra cleanup to find orphaned processes
            main_pgid = os.getpgid(os.getpid())
            for proc in psutil.process_iter():
                # if vizdoom or python process, belongs to this process group, and is not this (main) process
                # WARNING: this will kill concurrent training runs if they are launched from a common source (i.e. a bash script)
                # since they will share the same PGID
                if proc.name() in ('vizdoom', 'python') and os.getpgid(proc.pid) == main_pgid and proc.pid != os.getpid():
                    proc.kill()
        else:
            if self.waiting:
                for remote in self.remotes:
                    remote.recv()
            for remote in self.remotes:
                remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
