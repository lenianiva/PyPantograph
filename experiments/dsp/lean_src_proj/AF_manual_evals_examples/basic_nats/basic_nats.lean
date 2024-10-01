inductive UnaryNat : Type
| zero : UnaryNat
| succ : UnaryNat → UnaryNat
deriving Repr

#check UnaryNat
#check UnaryNat.zero
#check UnaryNat.succ
#check UnaryNat.succ UnaryNat.zero

-- 0
#eval UnaryNat.zero
-- 1
#eval (UnaryNat.succ UnaryNat.zero)
-- 2
#eval (UnaryNat.succ (UnaryNat.succ UnaryNat.zero))

-- open the namespace for UnaryNat
open UnaryNat
#check zero

def add_left : UnaryNat → UnaryNat → UnaryNat
| zero, n => n
| succ m', n => succ (add_left m' n)

#check add_left zero
#eval add_left zero zero
#eval add_left zero (succ zero)
#eval add_left (succ zero) zero

def add_right (m n : UnaryNat) : UnaryNat :=
  match n with
  | zero => m
  | succ n' => succ (add_right m n')

#eval add_right zero zero

-- todo add_right == add_left

infixl:65 "+" => add_left

#eval zero + zero -- add(0, 0)
-- a + b + c -> add(add(a, b), c) or add(a, add(b, c))

theorem add_zero_is_zero : zero + zero = zero := rfl

-- 0 + n = n
theorem zero_add_n_eq_n : ∀ n : UnaryNat, zero + n = n := by
  intro n
  rfl
  -- simp [add_left]
  -- rw [add_left]
  -- print the proof term for me please
  #print zero_add_n_eq_n

theorem zero_add_n_eq_n' (n : UnaryNat) : zero + n = n := by rfl
#print zero_add_n_eq_n'

-- n + 0 = n
theorem n_add_zero_eq_n : ∀ n : UnaryNat, n + zero = n := by
  intro n
  induction n with
  | zero => apply rfl
  -- | succ n' ih => rw [add_left]; rw [ih]
  | succ n' ih => rw [add_left, ih]
  #print n_add_zero_eq_n

-- comm, assoc, distrib, etc proofs? see software foundations?
