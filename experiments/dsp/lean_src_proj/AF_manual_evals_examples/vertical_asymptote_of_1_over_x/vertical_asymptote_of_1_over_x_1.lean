/-
-/
import Mathlib.Data.Real.Basic

-- define 1/x (reciprical) for reals
noncomputable def f (x : ℝ ):  ℝ := x⁻¹
#check f

-- unit test that f 1 = 1, f 2 = 1/2
theorem test_f1 : f 1 = 1 := by simp[f]
theorem test_f2 : f 2 = 2⁻¹ := by simp[f]
#print test_f1
#print test_f2

-- set_option pp.notation false
-- The limit of f x as x approaches c+ from the right is +infinity i.e., limit is unbounded from the right
-- i.e., lim_{x -> c+} f(x) = +infinity
def has_unbounded_limit_right (f: ℝ -> ℝ) (c : ℝ) : Prop :=
  ∀ M : ℝ, 0 < M → ∃ δ, 0 < δ ∧ ∀ x : ℝ, 0 < x - c ∧ x - c < δ → M < f x
#print has_unbounded_limit_right

theorem reciprocal_has_unbounded_limit_right : has_unbounded_limit_right f 0 := by
  unfold has_unbounded_limit_right
  intro M h_0_lt_M
  -- select delta that works since func is 1/x then anything less than 1/M will make f x be greater than M (so it should work)
  use M⁻¹
  -- TODO split (what did scott want with this, read)
  constructor
  . rwa [inv_pos]
  . -- consider any x with 0 < x - 0 < M⁻¹ but introduce both hypothesis 0 < x - 0 and x - 0 < M⁻¹
    intro x ⟨h_x_pos, h_x_lt_δ⟩
    -- rintro x ⟨h_x_pos, h_x_lt_δ⟩ -- TODO tomorrow, why did scott do this?
    -- rewrite both hypothesis using fact m - 0 = m
    rw [sub_zero] at h_x_pos h_x_lt_δ
    unfold f
    -- multiply both sides of h_x_lt_δ by x⁻¹ on the left using mul_lt_mul_right
    rwa [propext (lt_inv h_0_lt_M h_x_pos)]

-- state p f = 1 todo: https://proofassistants.stackexchange.com/questions/3800/given-some-proposition-in-lean-4-how-do-we-state-a-theorem-saying-that-we-want
