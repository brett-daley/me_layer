import gymnasium


class SeedWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.env = env
        self.seed = seed
        self.first_reset = True

    def reset(self, **kwargs):
        if self.first_reset:
            self.first_reset = False
            kwargs["seed"] = self.seed
            return self.env.reset(**kwargs)
        else:
            return self.env.reset(**kwargs)
