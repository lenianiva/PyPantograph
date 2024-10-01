/-

Specs:
- use reasonable amount mathlib, so it's ok to re-use the rationls from mathlib

Proofs
1. Showing A = {p ∈ ℚ | p² < 2} has no maximum element.
(2. p² = 2 has no rational solution, different file)

Thm: ∀p ∈ A, ∃ q ∈ A, p < q.
q = p + e

WTS: (p + e)² < 2
p² + 2pe + e²
p² + pe + pe + e²
intuitively want make e subject
p² + pe + e(p + e)
observe that p + e < 2 (lemma)
p² + pe + 2e < 2
p² + e(p + 2) < 2
e < 2 - p² / (p + 2)
-- plug e back into WTS to show it's true

-/

-- import Mathlib.Data.Rat.Basic
import Mathlib.Tactic

-- define in lean 4: A  = { p ∈ ℚ | p² < 2 }
-- def A : set ℚ :=
def A : Set ℚ := { p : ℚ | p ^ 2 < 2 }


-- Thm: ∀p ∈ A, ∃ q ∈ A, p < q. in lean 4
-- theorem exists_greater (p : ℚ) (hp : p ∈ A) : ∃ q ∈ A, p < q :=
-- begin
--   -- Proof would go here
-- end

-- theorem exists_greater : ∀ (p : ℚ), p ∈ A → ∃ (q : ℚ), q ∈ A ∧ p < q :=
theorem exists_greater : ∀ p ∈ A → ∃ (q : ℚ), q ∈ A ∧ p < q :=
sorry

theorem exists_gt_of_mem_nhds {α : Type*} [TopologicalSpace α]
    [LinearOrder α] [DenselyOrdered α] [OrderTopology α]
    [NoMaxOrder α] {a : α} {s : Set α}
    (h : s ∈ 𝓝 a) : ∃ b ∈ s, a < b := by
  obtain ⟨u, h₁, h₂⟩ := exists_Ico_subset_of_mem_nhds h (exists_gt _)
  obtain ⟨u', h₃, h₄⟩ := exists_between h₁
  exact ⟨u', h₂ ⟨h₃.le, h₄⟩, h₃⟩

/-- The set { x : ℚ | x*x < 2 } has no maximum -/
theorem no_max_sq_lt_two (x : ℚ) (hx : x * x < 2) :
    ∃ y, y * y < 2 ∧ x < y := by
  refine exists_gt_of_mem_nhds (IsOpen.mem_nhds ?_ hx)
  have : Continuous (fun t : ℚ => t * t) := by continuity
  exact this.isOpen_preimage (Set.Iio 2) isOpen_Iio
