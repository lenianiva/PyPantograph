"""Test for special tactics, such as have, let, etc."""
import pytest
from pantograph import Server
from pantograph.expr import *
from pantograph.server import TacticFailure
from .utils import apply_tactics


def test_simple_have():
    server = Server()
    state0 = server.goal_start("1 + (1: Nat) = 2")
    state1 = server.goal_tactic(state0, goal_id=0, tactic=TacticHave(branch="(2:Nat) = 1 + 1", binder_name="h"))
    state2 = server.goal_tactic(state1, goal_id=0, tactic="simp")
    state3 = server.goal_tactic(state2, goal_id=0, tactic="simp")
    assert state1.goals == [
        Goal(
            variables=[],
            target="2 = 1 + 1",
        ),
        Goal(
            variables=[Variable(name="h", t="2 = 1 + 1")],
            target="1 + 1 = 2",
        ),
    ]
    assert state2.goals == [
        Goal(
            variables=[Variable(name="h", t="2 = 1 + 1")],
            target="1 + 1 = 2",
        ),
    ]
    assert state3.is_solved

