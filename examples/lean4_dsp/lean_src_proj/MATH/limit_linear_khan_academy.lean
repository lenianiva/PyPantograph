/-
Theorem: lim_{x -> 3} f(x) = 6, f(x) = 2x if x ≠ 3 else x if x = 3
Proof:
WTS: ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| < δ → 0 < |f(x) - 6| < ε.
Consider any ε > 0.
Then let's start from what we want 0 < |f(x) - 6| < ε.
We want to have that be of the form 0 < |x - 3| < δ, and construct delta > 0.
Unfold the definition of f(x) when x ≠ 3:
  0 < |f(x) - 6| < ε
  0 < |2x - 6| < ε
  0 < |2(x - 3)| < ε
  0 < |2|*|x - 3| < ε
  0 < 2*|x - 3| < ε
  0 < |x - 3| < ε/2
So we can choose δ = ε/2.
Since ε > 0, δ > 0.
Qed.
-/
import Mathlib.Data.Real.Basic

-- define f, f(x) = 2x if x ≠ 3 else x if x = 3
def c : ℝ := 3
#eval c
def L : ℝ := 6

noncomputable def f (x : ℝ) : ℝ :=
  if x = 3.0 then x else 2.0 * x

-- Epsilon-delta definition of a limit for a function f at a point c approaching L
def has_limit_at (f : ℝ → ℝ) (L c : ℝ) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs (x - c) ∧ abs (x - c) < δ → abs (f x - L) < ε
#check abs
#eval abs 3

-- theorem: lim_{x -> 3} f(x) = 6
theorem lim_x_3_f_x_eq_6 : has_limit_at f L c := by
  intros ε ε_pos
  let δ := ε / 2
  have δ_pos : δ > 0 := div_pos ε_pos (by norm_num)
  use δ
  intro x
  intro h
  cases h with h1 h2
  cases (em (x = 3)) with x_eq_3 x_ne_3
  case inl =>
    rw [x_eq_3]
    simp
    exact ε_pos
  case inr =>
sorry
