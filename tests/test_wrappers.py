import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def wrapper_class(monkeypatch):
    class FakeEnvironment:
        def state(self):
            return self.raw_state

    minatar = types.ModuleType("minatar")
    minatar.Environment = FakeEnvironment
    monkeypatch.setitem(sys.modules, "minatar", minatar)
    monkeypatch.setitem(sys.modules, "domain", types.ModuleType("domain"))

    module_path = Path(__file__).parents[1] / "wrappers.py"
    spec = importlib.util.spec_from_file_location("wrappers_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MinatarWrapper


@pytest.mark.parametrize(
    "observation",
    [
        np.zeros((2, 2, 2)),
        np.full((2, 2, 2), 3),
        np.array(
            [
                [[0, 0], [1, 0]],
                [[0, 1], [1, 1]],
            ]
        ),
    ],
    ids=["all-zero", "constant-nonzero", "varied"],
)
def test_state_and_render_are_finite(wrapper_class, observation):
    wrapper = wrapper_class()
    wrapper.raw_state = observation

    state = wrapper._state()
    image_data = np.asarray(wrapper.render())

    assert np.isfinite(state).all()
    assert np.isfinite(image_data).all()


def test_constant_observations_return_neutral_state(wrapper_class):
    wrapper = wrapper_class()
    wrapper.raw_state = np.full((2, 2, 2), 3)

    np.testing.assert_array_equal(wrapper._state(), np.zeros((2, 2)))


def test_render_handles_zero_scaling_maximum(wrapper_class):
    wrapper = wrapper_class()
    wrapper._state = lambda: np.full((2, 2), -1.0)

    image_data = np.asarray(wrapper.render())

    assert np.isfinite(image_data).all()
    np.testing.assert_array_equal(image_data, np.zeros((200, 200), dtype=np.uint8))


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
