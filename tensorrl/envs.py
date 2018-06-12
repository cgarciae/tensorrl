
class TimeExpanded(object):

    def __init__(self, env, window):

        self.env = env
        self.window = window
        self.state = None

    def reset(self, *args, **kwargs):
        new_state = self.env.reset(*args, **kwargs)
        self.state = np.stack([new_state] * self.window, axis = -1)

        return self.state

    def step(self, *args, **kwargs):

        new_state, reward, done, info = self.env.step(*args, **kwargs)

        self.state[..., 1:] = self.state[..., :-1]
        self.state[..., 0] = new_state

        return self.state, reward, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class Physics(object):

    def __init__(self, env):

        self.env = env
        self.window = 2
        self.state = None

    def reset(self, *args, **kwargs):
        new_state = self.env.reset(*args, **kwargs)
        zeros = np.zeros_like(new_state)
        self.state = np.stack([new_state, zeros], axis = -1)

        return self.state

    def step(self, *args, **kwargs):

        new_state, reward, done, info = self.env.step(*args, **kwargs)
        old_state = self.state[..., 0]

        self.state[..., 0] = new_state
        self.state[..., 1] = new_state - old_state

        return self.state, reward, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)
