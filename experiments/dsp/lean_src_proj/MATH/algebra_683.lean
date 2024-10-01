-- {
--     "problem": "If $\\sqrt{2\\sqrt{t-2}} = \\sqrt[4]{7 - t}$, then find $t$.",
--     "level": "Level 4",
--     "type": "Algebra",
--     "solution": "We raise both sides to the fourth power, which is equivalent to squaring twice, in order to get rid of the radicals. The left-hand side becomes $$\\left(\\sqrt{2\\sqrt{t-2}}\\right)^4 = \\left(2\\sqrt{t-2}\\right)^2 = 4 \\cdot (t-2) = 4t-8.$$The right-hand side becomes $\\left(\\sqrt[4]{7-t}\\right)^4 = 7-t$. Setting them equal, $$4t-8 = 7-t \\quad\\Longrightarrow\\quad 5t = 15,$$and $t = \\boxed{3}$.  Checking, we find that this value does indeed satisfy the original equation."
-- }

import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Pow

noncomputable def a (t : ℝ) : ℝ := (2 * (t - 2) ^ (1 / 2)) ^ (1/2)
noncomputable def b (t : ℝ) : ℝ := (7 - t)^(1/4)

def valid_t (t : ℝ) : Prop :=
  a t = b t

theorem LHS_to_4 : ∀ t : ℝ, (a t) ^ 4 = 4 * t - 8 := by sorry
theorem RHS_to_4 : ∀ t : ℝ, (b t) ^ 4 = 7 - t := by sorry
theorem solution : valid_t 3 := by sorry
