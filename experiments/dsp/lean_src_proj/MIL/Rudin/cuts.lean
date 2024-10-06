/--
@[ext] structure Cut where
  S : Set ℚ

  not_all : S ≠ Set.univ
  nonempty : S.Nonempty

  lt_elem x y : x ∈ S → y < x → y ∈ S
  no_max  x : x ∈ S → ∃ y ∈ S, x < y
--/

-- structure == inductive, but structure has one constructor. it is all these things it's an and, it's like a named tuple or record in coq, structs in C
structure Cut where
  carrier: Set ℚ  -- carrier == α in Rudin, the actual cut
  not_empty: carrier ≠ {}  -- α ≠ ∅ (I)
  not_all: carrier ≠ Set.univ  -- alpha ≠ Set ℚ, carrier ≠ { _a | True} (I)
  p_less_q_in_carrier: ∀ p ∈ carrier, ∀ q : ℚ, q < p -> q ∈ carrier -- (II)
  p_less_q_in_carrier': (p : ℚ) -> p ∈ carrier -> (q : ℚ) -> q < p -> q ∈ carrier -- (III)
  always_exist_a_larger_element: ∀ p ∈ carrier, ∃ r ∈ carrier, p < r -- (III')
  if_q_not_in_carrier_then_grater_than_elem_in_carrier: ∀ p ∈ carrier, ∀ q : ℚ, (q ∉ carrier) → p < q

  -- Now let's debug/test the above structure we defined
  -- 1. let's proof some subset of Q is a (Dedikin) Cut
  -- 2. thm, rfl proof of that p_less_q_in_carrier' == p_less_q_in_carrier'
  -- 3. continue in Rudin!
  -- 4. Show (II) implies --> (IV) & (V)

def A : Set ℚ := { x | x < 3}
theorem A_is_a_cut : Cut d
{
  carrier := A,
  not_empty := sorry, -- Proof that A is not empty
  not_all := sorry, -- Proof that A is not the set of all rationals
  p_less_q_in_carrier := sorry, -- Proof of ∀ p ∈ A, ∀ q : ℚ, q < p -> q ∈ A
  p_less_q_in_carrier' := sorry, -- Alternative proof of the same property
  always_exist_a_larger_element := sorry, -- Proof of ∀ p ∈ A, ∃ r ∈ A, p < r
  if_q_not_in_carrier_then_grater_than_elem_in_carrier := sorry -- Proof of ∀ p ∈ A, ∀ q : ℚ, (q ∉ A) → p < q
}

  /--
  include private proofs discussed with scott
  --/
