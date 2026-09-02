import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

minatar = types.ModuleType("minatar")
minatar.Environment = object
sys.modules.setdefault("minatar", minatar)

from domain.config import games
from domain.task_gym import BudgetExhaustedError, GymTask


@pytest.mark.parametrize("game_name", ["minatar:breakout", "minatar:freeway"])
def test_minatar_activation_and_label_counts(game_name):
    game = games[game_name]

    assert len(game.i_act) == game.input_size == 100
    assert len(game.o_act) == game.output_size == 6
    assert game.in_out_labels[:2] == ["cell_0", "cell_1"]
    assert game.in_out_labels[-6:] == [
        "noop", "fire", "up", "down", "left", "right"
    ]
    assert len(game.in_out_labels) == game.input_size + game.output_size


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("i_act", [1], "input_size is 100, but i_act has 1 entries"),
        ("o_act", [1], "output_size is 6, but o_act has 1 entries"),
    ],
)
def test_gym_task_rejects_activation_count_mismatches(field, replacement, message):
    game = games["minatar:breakout"]._replace(**{field: replacement})

    with pytest.raises(ValueError, match=message):
        GymTask(game, paramOnly=True)


def make_fitness_task(budget, rewards):
    task = GymTask.__new__(GymTask)
    task.nReps = len(rewards)
    task.curr_eval = 0
    task.budget = budget
    task.testInd = lambda *args, **kwargs: rewards.pop(0)
    return task


def test_get_fitness_raises_when_budget_is_exhausted():
    rewards = [123.0]
    task = make_fitness_task(budget=0, rewards=rewards)

    with pytest.raises(BudgetExhaustedError, match="budget is exhausted"):
        task.getFitness(np.array([0.0]), np.array([0]), nRep=1)

    assert task.curr_eval == 0
    assert rewards == [123.0]


def test_get_fitness_averages_only_completed_evaluations():
    rewards = [2.0, 4.0, 1000.0]
    task = make_fitness_task(budget=2, rewards=rewards)

    fitness = task.getFitness(np.array([0.0]), np.array([0]), nRep=3)

    assert fitness == pytest.approx(3.0)
    assert task.curr_eval == 2
    assert rewards == [1000.0]
