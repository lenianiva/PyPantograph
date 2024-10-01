-- {
--     "problem": "Let $t(x) = \\sqrt{3x+1}$ and $f(x)=5-t(x)$. What is $t(f(5))$?",
--     "level": "Level 4",
--     "type": "Algebra",
--     "solution": "We first evaluate $f(5) = 5 -t(5) = 5-\\sqrt{5\\cdot3+1}=1$. Thus $t(f(5))=t(1)=\\sqrt{3\\cdot1 + 1}=\\boxed{2}$."
-- }

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.GroupPower.Order

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (3 * x + 1)
noncomputable def f (x : ℝ) : ℝ := 5 - t x

theorem solve_t_at_5: t 5 = 4 := by
  have h0 : Real.sqrt 4 ^ 2 = 4 := Real.sq_sqrt (Nat.ofNat_nonneg _)
  have h1 : 3 * 5 + 1 = 4^2 := by rfl
  have h2 : Real.sqrt (3 * 5 + 1) = Real.sqrt 4^2:= by sorry
  unfold t
  rw[h2, h0]

theorem solve_f_at_5: f 5 = 1 := by
  unfold f
  have h: t 5 = 4 := by apply solve_t_at_5
  rw[h]
  ring

theorem solve_t_f_at_5: t (f 5) = 2 := by
  unfold t
  have h0: f 5 = 1 := by apply solve_f_at_5
  have h1: 3 * 1 + 1 = 2^2 := by rfl
  have h2: Real.sqrt (3 * 1 + 1) = Real.sqrt 2^2 := by sorry
  have h3: Real.sqrt 2^2 = 2 := Real.sq_sqrt (Nat.ofNat_nonneg _)
  rw[h0, h2, h3]
