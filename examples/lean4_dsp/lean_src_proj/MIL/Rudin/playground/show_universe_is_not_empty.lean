/-
Show universe is not equal to empty

-/

import Mathlib.Algebra.Order.Field.Defs
import Mathlib.Logic.Basic
import Mathlib.Tactic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field.Defs

#synth Add Nat

#check Set
#check Set.univ -- Set.univ.{u_1} {α : Type u_1} : Set α
#print Set.univ -- {_a : True}
-- def Set.univ.{u_1} : {α : Type u_1} → Set α := fun {α} ↦ {_a | True}
#check ∅ -- {_a : False}
-- def Set.Empty.{u_1} : {α : Type u_1} → Set α := fun {α} ↦ {_a | False}
#synth EmptyCollection (Set Nat)
#check Set.instEmptyCollectionSet

-- theorem empty_is_not_universe: Set.univ ≠ ∅ := by
--   intro h

-- to pass implicit args
theorem universe_nat_is_not_empty: Set.univ (α := Nat) ≠ ∅ := by
  intro h
  change (fun x : Nat => True) = (fun x : Nat => False) at h
  -- we want to derive a new fact, so we use have
  have hh: True = False := congrFun h 0
  contradiction
  done

example : Set.univ (α := Nat) ≠ ∅ := by
  simp only [ne_eq, Set.univ_eq_empty_iff, not_isEmpty_of_nonempty, not_false_eq_true]
