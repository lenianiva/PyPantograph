"""Tests for the special cases."""
import pytest
from pantograph import Server
from pantograph.expr import *
from pantograph.server import TacticFailure


@pytest.mark.basic
def test_goal_start_with_ambiguous_type():
    server = Server()
    # Valid expression
    state = server.goal_start("(1:Nat) + 1 = 2")
    assert isinstance(state, GoalState) and len(state.goals) == 1
    state2 = server.goal_tactic(state, 0,  "rfl")
    assert state2.is_solved

    # Invalid expression
    state = server.goal_start("1 + 1 = 2")
    with pytest.raises(TacticFailure):
        server.goal_tactic(state, 0, "simp")
    
