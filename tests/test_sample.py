"""Tests for the special cases."""
import pytest
from pantograph import Server
from pantograph.expr import *
from pantograph.server import TacticFailure

special_cases = [
"""theorem mathd_numbertheory_552
  (f g h : ℕ+ → ℕ)
  (h₀ : ∀ x, f x = 12 * x + 7)
  (h₁ : ∀ x, g x = 5 * x + 2)
  (h₂ : ∀ x, h x = Nat.gcd (f x) (g x))
  (h₃ : Fintype (Set.range h)) :
  (∑ k in (Set.range h).toFinset, k) = 12 := by sorry
""",
"""theorem amc12_2000_p15 
  (f : ℂ → ℂ)
  (h₀ : ∀ x, f (x / 3) = x ^ 2 + x + 1)
  (h₁ : Fintype (f ⁻¹' {7})) :
  (∑ y in (f ⁻¹' {7}).toFinset, y / 3) = -1 / 9 := by sorry
""",
"""theorem sample (p q r x : ℂ)
  (h₀ : (x - p) _(x - q) = (r - p)_ (r - q)) :
  (x - r) _x = (x - r)_ (p + q - r) := by linear_combination h₀
"""
]

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

@pytest.mark.error
def test_jsondecode(minif2f_root):
    server = Server(imports=['Mathlib', 'Init'], project_path=minif2f_root)
    unit = server.load_sorry(special_cases[0])[0]
    goal_state, message = unit.goal_state, '\n'.join(unit.messages)
    assert goal_state is not None and 'error' not in message.lower() and len(goal_state.goals) == 1

    unit = server.load_sorry(special_cases[1])[0]
    goal_state, message = unit.goal_state, '\n'.join(unit.messages)
    assert goal_state is not None and 'error' not in message.lower() and len(goal_state.goals) == 1

@pytest.mark.error
def test_proof_with_warn_type(minif2f_root):
    server = Server(imports=['Mathlib', 'Init'], project_path=minif2f_root)
    unit = server.load_sorry(special_cases[2])[0]
    goal_state, message = unit.goal_state, '\n'.join(unit.messages)
    assert goal_state is None and 'error' not in message.lower()
