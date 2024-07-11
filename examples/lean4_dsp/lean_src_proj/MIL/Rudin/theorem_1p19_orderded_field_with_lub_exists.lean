/-
1.19 Theorem There exists an ordered field R which has the leaast-upper-bound property.
  Morevover, R contains Q as a subfield.

Plan of attack:
- import Q from mathlib
- import ordered field from mathlib
- define R as dedikind cuts like Rudin did in Step 1
- Prove it's an ordered Field

- TODO later:
  - define lub or from mathlib
  - then see what goes next from Rudin's thm 1.19 and continue proof
-/

import Mathlib.Algebra.Order.Field.Defs

-- #check LinearOrderedSemifield
#check Set
-- #check (Set)
#check Set.univ
-- #print Set.univ

-- structure == inductive, but structure has one constructor. it is all these things it's an and, it's like a named tuple or record in coq, structs in C
structure Cut where
  carrier: Set ℚ  -- carrier == α in Rudin, the actual cut
  not_empty: carrier ≠ {}  -- α ≠ ∅ (I)
  not_all: carrier ≠ Set.univ  -- alpha ≠ Set ℚ, carrier ≠ { _a | True} (I)
  -- Formalize: if q < p & p ∈ α ==> q ∈ α
  p_less_q_in_carrier: ∀ p ∈ carrier, ∀ q : ℚ, q < p -> q ∈ carrier -- (II)
  -- Formalize: (p : ℚ) -> p ∈ carrier -> (q : ℚ) -> q < p -> q ∈ carrier
  p_less_q_in_carrier': (p : ℚ) -> p ∈ carrier -> (q : ℚ) -> q < p -> q ∈ carrier -- (III)
  -- ∀ p ∈ α, ∃ r ∈ α s.t. p < r
  -- always_exist_a_larger_element: ∀ p ∈ carrier, ∃ r ∈ carrier -> p < r
  always_exist_a_larger_element: ∀ p ∈ carrier, ∃ r ∈ carrier, p < r -- (III')
  -- if p ∈ α and q ∉ α then p < q
  if_q_not_in_carrier_then_grater_than_elem_in_carrier: ∀ p ∈ carrier, ∀ q : ℚ, (q ∉ carrier) → p < q

  -- Now let's debug/test the above structure we defined

  -- 1. let's proof some subset of Q is a (Dedikin) Cut
  -- 2. thm, rfl proof of that p_less_q_in_carrier' == p_less_q_in_carrier'
  -- 3. continue in Rudin!
  -- 4. Show (II) implies --> (IV) & (V)

def A : Set ℚ := { x | x < 3 }
theorem A_is_a_cut : Cut :=
{
  carrier := A,
  not_empty := by
    intro h
    -- how to construct false (show the universe is not empty)

    sorry, -- Proof that A is not empty

  not_all := sorry, -- Proof that A is not the set of all rationals
  p_less_q_in_carrier := sorry, -- Proof of ∀ p ∈ A, ∀ q : ℚ, q < p -> q ∈ A
  p_less_q_in_carrier' := sorry, -- Alternative proof of the same property
  always_exist_a_larger_element := sorry, -- Proof of ∀ p ∈ A, ∃ r ∈ A, p < r
  if_q_not_in_carrier_then_grater_than_elem_in_carrier := sorry -- Proof of ∀ p ∈ A, ∀ q : ℚ, (q ∉ A) → p < q
}
