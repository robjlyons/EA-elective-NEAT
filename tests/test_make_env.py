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


@pytest.mark.parametrize(
    ("env_name", "loader_name"),
    [
        ("Classify_digits", "digit_raw"),
        ("Classify_mnist256", "mnist_256"),
        ("Classify_mnist784", "mnist_784"),
    ],
)
def test_classification_environment_uses_matching_dataset_loader(
        monkeypatch, make_env_module, env_name, loader_name):
    loader_calls = []
    datasets = {
        name: (object(), object())
        for name in ("digit_raw", "mnist_256", "mnist_784")
    }

    def loader(name):
        def load_dataset():
            loader_calls.append(name)
            return datasets[name]
        return load_dataset

    class FakeClassifyEnv(LegacySeedEnv):
        def __init__(self, train_set, target):
            super().__init__()
            self.dataset = (train_set, target)

    monkeypatch.setitem(
        sys.modules,
        "domain.classify_gym",
        module(
            ClassifyEnv=FakeClassifyEnv,
            digit_raw=loader("digit_raw"),
            mnist_256=loader("mnist_256"),
            mnist_784=loader("mnist_784"),
        ),
    )

    env = make_env_module.make_env(env_name)

    assert loader_calls == [loader_name]
    assert env.dataset == datasets[loader_name]


def test_unknown_classification_environment_raises_descriptive_error(
        monkeypatch, make_env_module):
    monkeypatch.setitem(
        sys.modules, "domain.classify_gym", module(ClassifyEnv=LegacySeedEnv)
    )

    with pytest.raises(
            ValueError,
            match="Unknown classification environment: 'Classify_unknown'",
    ):
        make_env_module.make_env("Classify_unknown")


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
