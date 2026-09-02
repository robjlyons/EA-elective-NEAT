import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

minatar = types.ModuleType("minatar")
minatar.Environment = object
sys.modules.setdefault("minatar", minatar)

from domain.config import games
from domain.task_gym import GymTask


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
