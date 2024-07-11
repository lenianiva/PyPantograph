/-
f(x) = m*x + c at x=x' and anything else o.w. (e.g., x)

WTS: lim_{x -> x'} f(x) = m*x' + c
-/
import Mathlib.Data.Real.Basic

-- Define the limit of a function at a point
def limit (f : ℝ → ℝ) (x' : ℝ) (l : ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ ∀ x : ℝ, 0 < abs (x - x') ∧ abs (x - x') < δ → abs (f x - l) < ε

-- Define the target function to reason about f(x) = m*x + c at x=x' and anything else o.w. (e.g., x)
noncomputable def lin (m c : ℝ) (x : ℝ) : ℝ := m*x + c
noncomputable def f (m c hole_x : ℝ) (x : ℝ) : ℝ := if x = hole_x then lin m c x else x

-- Prove the limit of a linear funtion with a hole at the point would be the lin value at the hole i.e., f(x) = m*x + c at x=x' is m*x' + c
theorem limit_of_lin_func_with_hole_eq_lin_func (m c limit_pt_x : ℝ) : limit (f m c hole_x) hole_x (lin m c hole_x) := by
  unfold limit
  intros ε ε_pos
  -- we want 0 < | f(x) - (m*x' + c) | < ε but in format 0 < | x - x' | < δ, so "invert f on both sides and put in δ format"
  -- we want 0 < | m*x + c - (m*x' + c) | < ε using def of f not at x'
  -- we want 0 < |m| * | x - x' | < ε --> 0 < | x - x' | < ε / |m| so δ = ε / |m|
  use ε / abs m
  apply And.intro
  .
