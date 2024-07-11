---- Example: define unary natural numbers

---- define unary nats
-- define unary natural numbers
inductive UnaryNat : Type
| Zero: UnaryNat
| Succ: UnaryNat -> UnaryNat
-- make unary nats printable
deriving Repr

-- define unary natural numbers
inductive MyNat : Type
| O: MyNat
| S: MyNat -> MyNat
-- make unary nats printable
deriving Repr
----

----
-- bring contents of unary nat into scope
open UnaryNat
-- bring contents of unary nat into scope
open MyNat
----

---- check types and evals
-- check type of unary nat, zero and succ
#check UnaryNat
#check UnaryNat.Zero
#check UnaryNat.Succ
#check UnaryNat.Succ UnaryNat.Zero
#check Succ (Succ Zero)
#eval UnaryNat.Zero
#eval UnaryNat.Succ UnaryNat.Zero
#eval UnaryNat.Succ (UnaryNat.Succ UnaryNat.Zero)
#eval Succ (Succ Zero)
#check O
#eval S (S O)
----

---- define addition for unary natural numbers
-- define addition for unary natural numbers (without explicit names in function declaration)
def add_left : UnaryNat -> UnaryNat -> UnaryNat
| Zero, n => n
| Succ m, n => Succ (add_left m n)

-- define addition for unary natural numbers (with explicit names in function declaration)
def add_left' (m n : UnaryNat) : UnaryNat :=
  match m with
  | Zero => n
  | Succ m' => Succ (add_left' m' n)

-- define addition infix notation
infixl:65 "+l" => add_left'

-- define right addition for unary natural numbers (without explicit names in function declaration)
def add_right : UnaryNat -> UnaryNat -> UnaryNat
| m, Zero => m
| m, Succ n => Succ (add_right m n)

-- define right addition for unary natural numbers (with explicit names in function declaration)
def add_right' (m n : UnaryNat) : UnaryNat :=
  match n with
  | Zero => m
  | Succ n' => Succ (add_right' m n')

-- define right addition infix notation
infixl:65 "+r " => add_right'
---

---- evals for addition
-- eval addition for unary natural numbers left and right
#eval Zero +l Zero
#eval Zero +l (Succ Zero)
#eval (Succ Zero) +l (Succ Zero)
#eval (Succ (Succ Zero)) +r (Succ Zero)
---

---- theorem show non inductive case of addition
-- theorem left addition, 0 + n = n (not inductive proof)
theorem add_left_zero_plus_n_eq_n (n : UnaryNat) : Zero +l n = n := by rfl
-- theorem left addition, 0 + n = n (not inductive proof) with forall statements
theorem add_left_zero_plus_n_eq_n' : Zero +l n = n := by intros; rfl
theorem add_left_zero_plus_n_eq_n'' : Zero +l n = n := by
  intros
  rfl
-- theorem right addition, n + 0 = n (not inductive proof)
theorem add_right_n_plus_zero_eq_n (n : UnaryNat) : n +r Zero = n := by rfl
-- theorem right addition, n + 0 = n (not inductive proof) with forall statements
theorem add_right_n_plus_zero_eq_n' : n +r Zero = n := by intros; rfl
theorem add_right_n_plus_zero_eq_n'' : n +r Zero = n := by
  intros
  rfl
----

---- theorem show inductive case of addition
-- theorem left addition, n + 0 = n (inductive proof)
theorem add_left_n_plus_zero_eq_n (n : UnaryNat) : n +l Zero = n := by
  induction n with
  | Zero => rfl
  | Succ n' ih => simp [add_left', ih]
-- theorem left addition, n + 0 = n (inductive proof) with forall and use inductive hypothesis explicitly
theorem add_left_n_plus_zero_eq_n' : ∀ (n : UnaryNat), n +l Zero = n := by
  intros n
  induction n with
  | Zero => rfl
  | Succ n' ih => simp [add_left']; assumption
