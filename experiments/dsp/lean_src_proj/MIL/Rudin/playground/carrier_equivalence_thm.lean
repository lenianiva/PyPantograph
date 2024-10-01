/-
Goal:
  -- (p : ℚ) -> p ∈ carrier -> (q : ℚ) -> q < p -> q ∈ carrier
  p_less_q_in_carrier: ∀ p ∈ carrier, ∀ q : ℚ, q < p -> q ∈ carrier
  p_less_q_in_carrier': (p : ℚ) -> p ∈ carrier -> (q : ℚ) -> q < p -> q ∈ carrier
  -- TODO: rfl proof of that
-/

import Mathlib.Algebra.Order.Field.Defs

-- def carrier (P : ℚ -> Prop) : Set ℚ := {x | P x}
-- def carrier: Set ℚ := {x}

-- def carrier: Set ℚ
-- #check carrier

-- Define the type with name "p_less_q_in_carrier" equal to ∀ p ∈ carrier, ∀ q : ℚ, q < p -> q ∈ carrier
-- def p_less_q_in_carrier: Type := ∀ p ∈ carrier, ∀ q : ℚ, q < p -> q ∈ carrier
