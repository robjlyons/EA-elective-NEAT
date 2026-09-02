import importlib.util
import sys
import types
from pathlib import Path

import pytest


class LegacySeedEnv:
    def __init__(self, *args, **kwargs):
        self.seed_received = None

    def seed(self, seed):
        self.seed_received = seed


class ResetSeedEnv:
    def __init__(self, *args, **kwargs):
        self.seed_received = None

    def reset(self, *, seed=None):
        self.seed_received = seed


def module(**attributes):
    fake_module = types.ModuleType("fake")
    for name, value in attributes.items():
        setattr(fake_module, name, value)
    return fake_module


@pytest.fixture
def make_env_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "wrappers", module(MinatarWrapper=LegacySeedEnv)
    )
    module_path = Path(__file__).parents[1] / "domain" / "make_env.py"
    spec = importlib.util.spec_from_file_location("make_env_under_test", module_path)
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    return loaded_module


@pytest.mark.parametrize(
    ("env_name", "module_name", "attributes"),
    [
        ("BipedalWalker-v2", "domain.bipedal_walker",
         {"BipedalWalker": LegacySeedEnv}),
        ("VAERacing-v0", "domain.vae_racing", {"VAERacing": LegacySeedEnv}),
        ("Classifydigits", "domain.classify_gym",
         {"ClassifyEnv": LegacySeedEnv, "digit_raw": lambda: ([], [])}),
        ("CartPoleSwingUp-v0", "domain.cartpole_swingup",
         {"CartPoleSwingUpEnv": LegacySeedEnv}),
    ],
)
def test_custom_environment_categories_seed_the_created_environment(
        monkeypatch, make_env_module, env_name, module_name, attributes):
    monkeypatch.setitem(sys.modules, module_name, module(**attributes))

    env = make_env_module.make_env(env_name, seed=7)

    assert env.seed_received == 7


@pytest.mark.parametrize("env_name", ["Acrobot-v1", "FakeBulletEnv-v0"])
def test_gym_environment_categories_use_modern_reset_seeding(monkeypatch,
                                                             make_env_module,
                                                             env_name):
    if "Bullet" in env_name:
        kuka_module = module()
        bullet_module = module(kukaGymEnv=kuka_module)
        monkeypatch.setitem(sys.modules, "pybullet", module())
        monkeypatch.setitem(
            sys.modules, "pybullet_envs", module(bullet=bullet_module)
        )
        monkeypatch.setitem(
            sys.modules, "pybullet_envs.bullet", bullet_module
        )
        monkeypatch.setitem(
            sys.modules, "pybullet_envs.bullet.kukaGymEnv", kuka_module
        )

    monkeypatch.setitem(
        sys.modules, "gym", module(make=lambda unused_name: ResetSeedEnv())
    )

    env = make_env_module.make_env(env_name, seed=11)

    assert env.seed_received == 11


def test_minatar_environment_receives_seed_during_construction(
        monkeypatch, make_env_module):
    constructor_arguments = {}

    def fake_wrapper(env_name, **kwargs):
        constructor_arguments.update(env_name=env_name, **kwargs)
        return object()

    monkeypatch.setattr(make_env_module, "MinatarWrapper", fake_wrapper)

    make_env_module.make_env("minatar:breakout", seed=13)

    assert constructor_arguments == {
        "env_name": "breakout",
        "sticky_action_prob": 0.0,
        "random_seed": 13,
    }
