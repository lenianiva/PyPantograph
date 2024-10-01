-- original: https://gist.github.com/brando90/10b84e485ea5d4ec8d828da8ddd20d47

-- define unary natural numbers
inductive UnaryNat : Type
| Zero: UnaryNat
| Succ: UnaryNat -> UnaryNat

-- make unary nats printable
deriving Repr

-- check the types
#check UnaryNat
#check UnaryNat.Zero

-- let's construct some unary nats
#eval UnaryNat.Zero
#eval UnaryNat.Succ UnaryNat.Zero
#eval UnaryNat.Succ (UnaryNat.Succ UnaryNat.Zero)

-- bring contents of namespace Unary_Nat into scope
open UnaryNat

-- define addition for unary nats
def add_left : UnaryNat -> UnaryNat -> UnaryNat
| Zero, m => m
| Succ n, m => Succ (add_left n m)

-- define infix notation for addition
infixl:65 " + " => add_left

-- test that addition works
#eval add_left Zero Zero
#eval add_left Zero (Succ Zero)
#eval add_left (Succ Zero) Zero

-- write a unit test as a theorem or example that checks addition works (e.g., 1+1=2)
theorem add_left_test : add_left (Succ Zero) (Succ Zero) = (Succ (Succ Zero)) := rfl
theorem add_left_infix_test: (Succ Zero) + (Succ Zero) = (Succ (Succ Zero)) := rfl

-- theorem for addition (the none inductive one)
theorem add_zero_plus_n_eq_n (n : UnaryNat) : Zero + n = n := rfl
-- theorem for addition using forall statements (the none inductive one)
theorem add_zero_plus_n_eq_n' : ∀ (n : UnaryNat), Zero + n = n := by
  intro n
  rfl
-- theorem for addition unfolding the definition of unary addition (none inductive)
theorem add_zero_plus_n_eq_n'' (n : UnaryNat) : Zero + n = n := by simp [add_left]

-- theorem for addition (the inductive one)
theorem add_n_plus_zero_eq_n (n : UnaryNat) : n + Zero = n := by
  induction n with
  | Zero => rfl
  | Succ n' ih => simp [add_left, ih]

-- theorem for addition using forall statements (the inductive one)
theorem add_n_plus_zero_eq_n' : ∀ (n : UnaryNat), n + Zero = n := by
  intro n
  induction n with
  | Zero => rfl
  | Succ n' ih => simp [add_left, ih]
