from wrappers import MinatarWrapper


def make_env(env_name, seed=-1, render_mode=False):
    seeded_on_creation = False
    # -- Bullet Environments ------------------------------------------- -- #
    if "Bullet" in env_name:
        import pybullet as p  # pip install pybullet
        import pybullet_envs
        import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

    # -- Bipedal Walker ------------------------------------------------ -- #
    if (env_name.startswith("BipedalWalker")):
        if (env_name.startswith("BipedalWalkerHardcore")):
            import Box2D
            from domain.bipedal_walker import BipedalWalkerHardcore
            env = BipedalWalkerHardcore()
        elif (env_name.startswith("BipedalWalkerMedium")):
            from domain.bipedal_walker import BipedalWalker
            env = BipedalWalker()
            env.accel = 3
        else:
            from domain.bipedal_walker import BipedalWalker
            env = BipedalWalker()


    # -- VAE Racing ---------------------------------------------------- -- #
    elif (env_name.startswith("VAERacing")):
        from domain.vae_racing import VAERacing
        env = VAERacing()


    # -- Classification ------------------------------------------------ -- #
    elif (env_name.startswith("Classify")):
        from domain.classify_gym import ClassifyEnv
        if env_name.endswith("digits"):
            from domain.classify_gym import digit_raw
            trainSet, target = digit_raw()

        if env_name.endswith("mnist256"):
            from domain.classify_gym import mnist_256
            trainSet, target = mnist_256()

        env = ClassifyEnv(trainSet, target)

        # -- Cart Pole Swing up -------------------------------------------- -- #
    elif (env_name.startswith("CartPoleSwingUp")):
        from domain.cartpole_swingup import CartPoleSwingUpEnv
        env = CartPoleSwingUpEnv()
        if (env_name.startswith("CartPoleSwingUp_Hard")):
            env.dt = 0.01
            env.t_limit = 200

    # +== EA-elective-NEAT =============================================================================================
    elif env_name.startswith("minatar:"):
        env_name = env_name.split(':')[1]
        random_seed = seed if seed >= 0 else 0
        env = MinatarWrapper(env_name, sticky_action_prob=.0,
                             random_seed=random_seed)
        seeded_on_creation = seed >= 0
    # =================================================================================================================+

    # -- Other  -------------------------------------------------------- -- #
    else:
        import gym
        env = gym.make(env_name)

    if seed >= 0 and not seeded_on_creation:
        # Gym used ``env.seed`` before reset accepted a seed keyword.  Prefer
        # the legacy method when it is available, while remaining compatible
        # with newer Gym environments that only expose reset(seed=...).
        seed_method = getattr(env, "seed", None)
        if callable(seed_method):
            seed_method(seed)
        else:
            env.reset(seed=seed)

    return env
