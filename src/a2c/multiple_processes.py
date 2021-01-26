from multiprocessing import Process, Pipe


def worker(remote, parent_remote, env_fn_wrapper):
    """This function gets run in separate worker processes.

    :param remote: Sort of a channel which is sends and
    receives commands by the main process.
    :param parent_remote:
    :param env_fn_wrapper: Objet returns an environment
    object when calling `x` method on it. `x` is the
    environment constructor.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            if done:
                observation = env.reset()
            remote.send((observation, reward, done, info))
        elif cmd == 'reset':
            observation = env.reset()
            remote.send(observation)
        elif cmd == 'reset_task':
            observation = env.reset_task()
            remote.send(observation)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_id':
            remote.send(env.spec.id)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing
    tries to use pickle).
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, observation):
        import pickle
        self.x = pickle.loads(observation)


class SubprocVecEnv:
    def __init__(self, env_fns):
        """
        The master that lives in the main process. The idea behind is that
        it is going to act like a vector of environments.

        env_fns: function that creates an environment
        """
        self.closed = False
        nenvs = len(env_fns)

        # We create the remotes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_id', None))
        self.env_id = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        # print("Infos:", infos)
        # for done, info in zip(dones, infos):
        #     if done:
        #         # print("Total reward:", info['reward'], "Num steps:", info['episode_length'])
        #         print("Returned info:", info, "Done:", done)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)