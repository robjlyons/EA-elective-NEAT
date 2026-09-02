import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def wrapper_class(monkeypatch):
    class FakeEnvironment:
        pass

    minatar = types.ModuleType("minatar")
    minatar.Environment = FakeEnvironment
    monkeypatch.setitem(sys.modules, "minatar", minatar)
    monkeypatch.setitem(sys.modules, "domain", types.ModuleType("domain"))

    module_path = Path(__file__).parents[1] / "wrappers.py"
    spec = importlib.util.spec_from_file_location("wrappers_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MinatarWrapper


def make_wrapper(wrapper_class):
    wrapper = wrapper_class()
    wrapper.num_actions = lambda: 6
    wrapper._state = lambda: np.array([[1, 2]])
    wrapper.actions_received = []

    def act(action):
        wrapper.actions_received.append(action)
        return 1, False

    wrapper.act = act
    return wrapper


def test_step_uses_probability_distribution_to_select_action(wrapper_class):
    wrapper = make_wrapper(wrapper_class)

    state, reward, done, info = wrapper.step([0, 0, 1, 0, 0, 0])

    assert wrapper.actions_received == [2]
    assert isinstance(wrapper.actions_received[0], int)
    np.testing.assert_array_equal(state, [1, 2])
    assert (reward, done, info) == (1, False, {})


def test_step_accepts_a_valid_scalar_discrete_action(wrapper_class):
    wrapper = make_wrapper(wrapper_class)

    wrapper.step(np.int64(4))

    assert wrapper.actions_received == [4]
    assert isinstance(wrapper.actions_received[0], int)


@pytest.mark.parametrize("action", [-1, 6, 1.5])
def test_step_rejects_invalid_scalar_discrete_action(wrapper_class, action):
    wrapper = make_wrapper(wrapper_class)

    with pytest.raises(ValueError):
        wrapper.step(action)

    assert wrapper.actions_received == []


def test_make_env_uses_supported_minatar_seed_argument(monkeypatch):
    constructor_arguments = {}

    class FakeMinatarWrapper:
        def __init__(self, env_name, sticky_action_prob, seed):
            constructor_arguments.update(
                env_name=env_name,
                sticky_action_prob=sticky_action_prob,
                seed=seed,
            )

    wrappers = types.ModuleType("wrappers")
    wrappers.MinatarWrapper = FakeMinatarWrapper
    monkeypatch.setitem(sys.modules, "wrappers", wrappers)

    module_path = Path(__file__).parents[1] / "domain" / "make_env.py"
    spec = importlib.util.spec_from_file_location("make_env_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    environment = module.make_env("minatar:breakout")

    assert isinstance(environment, FakeMinatarWrapper)
    assert constructor_arguments == {
        "env_name": "breakout",
        "sticky_action_prob": 0.0,
        "seed": 0,
    }
