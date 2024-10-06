import Mathlib.Algebra.Order.Field.Defs

#check LinearOrderedSemifield
#check Set
#check (Set)

inductive foo : Nat → Type
| ok n : foo n
def f : ∀ (n : ℕ), foo n := fun n => .ok n
#check (f)

inductive u_nat: Type
| zero: u_nat
| succ: u_nat -> u_nat
