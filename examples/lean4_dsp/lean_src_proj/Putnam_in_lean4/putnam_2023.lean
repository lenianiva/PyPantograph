/-
https://kskedlaya.org/putnam-archive/2023.pdf
-/

/- A1
  For a positive integer n,
  let fn(x) =vcos(x) cos(2x) cos(3x)··· cos(nx).
  Find the smallest n such that | f''_n(0)| > 2023.
-/

/-
(n : Nat), f_n(x) = cos(x) * ... * cons(nx)
-/
-- import Mathlib.Tactic
-- import Mathlib.Algebra.BigOperators.Basic

-- open BigOperators

-- /- likely pain points
--   - type coercions (arithmetic on ℕ vs ℝ vs ℚ etc
--     - having operations that work on both nats and reals
--   - having to put the answer in the question or how to restrict the concreteness of the answer needed for the proof to work
--     - possible solutions
--       - 1. writing the answer in the theorem and making it clear the limitations of the benchmark in the paper e.g., it doesn't mean it is a putnam fellow due to simplicity of prover
--         - annotated clearly problems that have limitations
--       - 2. restricting the logic (e.g., SMT solver) or writing constraints on the problem?
-- -/
-- #check Finset.Icc
-- -- #eval decide (0 ∈ Finset.range 3)
-- -- #eval decide (3 ∈ Finset.Icc 1 3)

-- def smallest (P : ℕ → Prop) (n : ℕ) := P n ∧ ∀ k, k < n → ¬ P k

-- #check Real.cos
-- example
--     (_ : f = fun n x => ∏ k : ℕ in Finset.Icc 1 n, Real.cos (k * x))
--     (second_deriv : (ℝ → ℝ) -> (ℝ → ℝ))
--     (abs : ℝ → ℝ)
--     (_ : P = fun n => abs (second_deriv (f n) 0) > 2023)
--     : ∃ n, smallest P n :=
--   by sorry

-- example
--     (P : ℕ → Prop)
--     : { x | P x } :=
--     by sorry


-- Praneeth Kolichala
-- Today, we iterated the product rule (deriv_mul ) in order to get a rule for the derivative of a product over a set (deriv_prod). This is the stuff we actually got to:
import Mathlib.Data.Complex.Exponential
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.NumberTheory.Bernoulli

open BigOperators
open Classical

-- Putnam analysis question
-- Compute the second derivative of
-- d^2/dx^2 cos(x)cos(2x)...cos(nx) at x=0

section product_rule
variable {𝕜 : Type u} [NontriviallyNormedField 𝕜] {x : 𝕜} {𝔸 : Type u_2} [NormedCommRing 𝔸] [NormedAlgebra 𝕜 𝔸]

noncomputable def deriv_if (p : Prop) [Decidable p] (f : 𝕜 → 𝔸) : 𝕜 → 𝔸 :=
  if p then deriv f else f

theorem deriv_if_pos {p : Prop} [Decidable p] {f : 𝕜 → 𝔸} (h : p) :
    deriv_if p f = deriv f := if_pos h

theorem deriv_if_neg {p : Prop} [Decidable p] {f : 𝕜 → 𝔸} (h : ¬p) :
    deriv_if p f = f := if_neg h

theorem differentiable_prod (S : Finset α) (f : α → 𝕜 → 𝔸) (x : 𝕜)
    (hdiff : ∀ i, DifferentiableAt 𝕜 (f i) x) :
    DifferentiableAt 𝕜 (fun x => ∏ i in S, f i x) x := by
  induction S using Finset.induction_on
  case empty => simp
  case insert i S hi h =>
  · simp [hi]
    exact (hdiff i).mul h

theorem deriv_prod [DecidableEq α] (S : Finset α) (f : α → 𝕜 → 𝔸)
    (hdiff : ∀ i : α, Differentiable 𝕜 (f i)) :
    deriv (fun z => ∏ i in S, f i z) =
      fun x => ∑ i in S, ∏ j in S, deriv_if (i = j) (f j) x := by
  funext x
  induction S using Finset.induction_on
  · simp
  case insert t S ht ih =>
    simp only [Finset.prod_insert ht]
    rw [deriv_mul]
    · rw [ih, Finset.sum_insert ht, deriv_if_pos rfl, Finset.mul_sum]
      have : ∀ j ∈ S, t ≠ j := by rintro j hj rfl; contradiction
      congr 1
      · congr 1
        apply Finset.prod_congr rfl
        intro j hj
        rw [deriv_if_neg (this j hj)]
      · apply Finset.sum_congr rfl
        intro i hi
        rw [deriv_if_neg (this i hi).symm]
    · exact hdiff t x
    · apply differentiable_prod; intro i; exact hdiff i x

end product_rule

Praneeth Kolichala
  21 minutes ago
First, I proved some simple lemmas about the derivatives of some functions. These are all just a couple lines but ideally there would be a tactic/CAS system to discharge any of them automatically.

-- Ideally these would be done by a tactic/cas system
section deriv_computation
/-- cos'((k + 1)x) = -sin((k+1)x) * (k + 1) -/
lemma cos_kx_hasDeriv (k x : ℝ) :
    HasDerivAt (fun z => Real.cos ((k + 1) * z)) (-Real.sin ((k + 1) * x) * (k + 1)) x := by
  apply HasDerivAt.cos
  simpa using HasDerivAt.const_mul (k + 1) (hasDerivAt_id' x)

/-- Same as above but using `deriv` instead of `HasDerivAt` -/
lemma deriv_cos_kx (k : ℝ) :
    deriv (fun z => Real.cos ((k + 1) * z)) = fun z => -Real.sin ((k + 1) * z) * (k + 1) := by
  funext x
  exact (cos_kx_hasDeriv k x).deriv

/-- cos''((k + 1)x) = -cos((k+1)x)*(k+1)^2 -/
lemma cos_kx_deriv_hasDeriv (k x : ℝ) :
    HasDerivAt (fun z => -Real.sin ((k + 1) * z) * (k + 1)) (-Real.cos ((k + 1) * x) * (k + 1) * (k + 1)) x := by
  apply HasDerivAt.mul_const
  simp only [neg_mul]
  apply HasDerivAt.neg
  apply HasDerivAt.sin
  simpa using HasDerivAt.const_mul (k + 1) (hasDerivAt_id' x)

/-- Same as above but using `deriv` instead of `HasDerivAt` -/
lemma cos_kx_deriv_deriv (k : ℝ) :
    (deriv <| deriv <| fun z => Real.cos ((k + 1) * z)) = fun x => -Real.cos ((k + 1) * x) * (k + 1) * (k + 1) := by
  funext x
  simp only [deriv_cos_kx]
  exact (cos_kx_deriv_hasDeriv k x).deriv
end deriv_computation


Praneeth Kolichala
  19 minutes ago
Next, we define some helper functions and explicitly compute the derivative of cos(x)cos(2x)...cos(nx). Notice how much nicer deriv_if makes this (compared to explicitly writing out all the cases):
/-- The product cos(x)cos(2x)⋯cos(nx) -/
noncomputable def cos_prod (n : ℕ) (x : ℝ) : ℝ :=
  ∏ i in Finset.range n, Real.cos ((i+1 : ℝ) * x)

/-- The product cos(x)cos(2x)⋯cos'(ix)⋯cos'(jx)⋯cos(nx)
  This is used when computing the derivative of cos_prod
  If i=j, then it will become
  cos(x)cos(2x)⋯cos''(ix)⋯cos(nx) -/
noncomputable def cos_prod_ij (n i j : ℕ) (x : ℝ) : ℝ :=
  ∏ k in Finset.range n, (deriv_if (j = k) <| deriv_if (i = k) <| fun z => Real.cos ((k + 1) * z)) x

theorem cos_prod_deriv (n : ℕ) :
    deriv (fun x => cos_prod n x) = fun x =>
      ∑ j in Finset.range n,
      ∏ k in Finset.range n,
      (deriv_if (j = k) <| fun z => Real.cos ((k + 1) * z)) x := by
  ext x
  simp only [cos_prod]
  rw [deriv_prod]
  intro i z
  refine (cos_kx_hasDeriv _ _).differentiableAt

theorem cos_prod_snd_deriv (n : ℕ) :
  deriv^[2] (fun x => cos_prod n x) = fun x =>
      ∑ i in Finset.range n,
      ∑ j in Finset.range n,
      cos_prod_ij n i j x := by
  simp only [Nat.iterate, cos_prod_deriv, cos_prod_ij]
  ext x
  -- A differentiability lemma about deriv_if
  have differentiable : ∀ {p : Prop} [Decidable p] (j : ℕ), Differentiable ℝ (deriv_if p <| fun z => Real.cos ((j + 1) * z)) := by
    intro p _ j
    simp only [deriv_if, deriv_cos_kx]
    split_ifs
    · intro z; refine (cos_kx_deriv_hasDeriv _ _).differentiableAt
    · intro z; refine (cos_kx_hasDeriv _ _).differentiableAt
  rw [deriv_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [deriv_prod]
  -- Close out differentiability conditions
  · intro; apply differentiable
  · intro i _; apply differentiable_prod; intro j; apply differentiable


Praneeth Kolichala
  18 minutes ago
We prove two important results about cos_prod_ij: it is 0 except when i=j, in which case it is -(i+1)^2:

theorem cos_prod_ij_eq_zero (i j : ℕ) (hi : i ∈ Finset.range n) (hij : i ≠ j) :
    cos_prod_ij n i j 0 = 0 := by
  apply Finset.prod_eq_zero hi
  rw [deriv_if_neg hij.symm, deriv_if_pos rfl, deriv_cos_kx]
  simp

theorem cos_prod_ij_eq_sq (i : ℕ) (hi : i ∈ Finset.range n) :
    cos_prod_ij n i i 0 = -((i : ℝ) + 1)^2 := by
  rw [cos_prod_ij, Finset.prod_eq_single i]
  · simp only [deriv_if_pos, cos_kx_deriv_deriv]
    simp; ring
  · intro j _ hij
    simp [deriv_if_neg hij.symm]
  · intro; contradiction


Praneeth Kolichala
  17 minutes ago
Finally, the main result:
theorem cos_prod_snd_deriv_eq(n : ℕ) :
    deriv^[2] (fun x => cos_prod n x) 0 = -∑ i in Finset.range n, (i+1)^2 := by
  calc
    deriv^[2] (fun x => cos_prod n x) 0 =
        ∑ i in Finset.range n, ∑ j in Finset.range n, cos_prod_ij n i j 0 := by rw [cos_prod_snd_deriv]
    _ = ∑ i in Finset.range n, cos_prod_ij n i i 0 := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [Finset.sum_eq_single i]
      intro j _ hij
      rw [cos_prod_ij_eq_zero i j hi hij.symm]
      intro; contradiction
    _ = ∑ i in Finset.range n, -((i : ℝ) + 1)^2 := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [cos_prod_ij_eq_sq i hi]
    _ = _ := by simp
Also sent to the channel


Praneeth Kolichala
  10 minutes ago
Pastebin with full code: https://pastebin.com/56jSmyk8
