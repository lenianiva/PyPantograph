inductive UnaryNat : Type :=
  | Zero : UnaryNat
  | Succ : UnaryNat -> UnaryNat -- Succ Zero 1, Succ (Succ Zero) 2
deriving Repr

-- P := n + Zero = n
-- ∀ n : ℕ, n + 0 = n
-- P Zero
-- P n' --> P (Succ n')
-- P n
-- UnaryNat.induct : P Zero -> (∀ n : UnaryNat, P n -> P (Succ n)) -> ∀ n : UnaryNat, P n
-- show me the induction principle for UnaryNat
#print UnaryNat.rec

#eval UnaryNat.Zero
#eval UnaryNat.Succ UnaryNat.Zero
#check UnaryNat.Zero
#check UnaryNat.Succ

-- create a variable that holds a unary zero
def unaryZero : UnaryNat := UnaryNat.Zero
#eval unaryZero

open UnaryNat

#eval Zero
#eval Succ Zero
#eval Succ (Succ Zero)

def add_left : UnaryNat -> UnaryNat -> UnaryNat
  | Zero, n => n  -- 0 + n => n
  | Succ m, n => Succ (add_left m n)  -- (m + 1) + n => (m + n) + 1

#eval add_left Zero Zero
#eval add_left Zero (Succ Zero)
#eval add_left (Succ Zero) Zero
#eval add_left (Succ Zero) (Succ Zero)
#check add_left
#check add_left Zero
#check add_left Zero Zero

-- def add_left' (m n : UnaryNat) : UnaryNat :=
--   match m with
--   | Zero => n
--   | Succ m' => Succ (add_left' m' n)

-- -- ∀ n : ℕ, 0 + n = 0
-- -- consider some natural number n
-- -- 0 + n = n is zero because of our definition add_left. Qed.
-- theorem zero_plus_zero : ∀ n : UnaryNat, add_left Zero n = n := by
--   intro n -- consider some n
--   rw [add_left]
--   -- unfold add_left
--   -- rfl

-- define add_left with infix notation
infixl:65 " + " => add_left

-- theorem zero_plus_zero' : ∀ n : UnaryNat, Zero + n = n := by
--   intro n -- consider some n
--   rw [add_left]
--   -- unfold add_left
--   -- rfl

-- ∀ n : ℕ, 0 + n = 0
-- consider some natural number n
-- 0 + n = n is zero because of our definition add_left. Qed.
theorem zero_plus_zero : ∀ n : UnaryNat, Zero + n = n := by
  intro n -- consider some n
  rw [add_left]
  -- unfold add_left
  -- rfl

theorem n_plus_zero : ∀ n : UnaryNat, n + Zero = n := by
  intros n -- consider some n of type UnaryNat
  -- do induction on n and call the induction hypothesis IH
  induction n with -- 0, S n
    -- P 0, 0 + 0 = 0 is true by add_left
    | Zero => rw [add_left]
    -- P n' -> P (S n')
    | Succ n' IH => rw [add_left, IH]
    -- rw [add_left]; rw [IH]
    -- rw [add_left, IH]
    -- simp [add_left, IH]
    -- simp [add_left]; rw [IH]
    -- simp [add_left]; assumption
      -- Proof state
      -- n' : UnaryNat
      -- IH : n' + Zero = n'
      -- ⊢ Succ n' + Zero = Succ n' <-- WTF
      -- Succ n' + Zero = Succ n'
      -- rewrite with add_left: Succ m + n => Succ (m + n)
      -- Succ (n' + Zero)

-- https://github.com/marshall-lee/software_foundations/blob/master/lf/Induction.v
-- n + (m + 1) = (n + m) + 1
theorem plus_n_Sm : ∀ n m : UnaryNat, n + Succ m = Succ (n + m) := by
  intros n m
  induction n with
    | Zero => rw [add_left, add_left]
    | Succ n' IH => rw [add_left, IH, add_left]
    -- show me the proof term

theorem plus_n_Sm' (n m : UnaryNat) : n + Succ m = Succ (n + m) := by apply plus_n_Sm

-- commutativity: n + m = m + n
theorem add_comm: ∀ n m : UnaryNat, n + m = m + n := by
  intros n m
  induction n with
    | Zero => rw [n_plus_zero, zero_plus_zero]
    | Succ n' IH => rw [plus_n_Sm, add_left, IH]

-- associativity: n + (m + p) = (n + m) + p.
theorem add_assoc : ∀ n m p : UnaryNat, n + (m + p) = (n + m) + p := by
  intros n m p
  induction n with
    | Zero => rw [add_left, add_left] -- simp [add_left]
    | Succ n' IH => rw [add_left, IH]; rw [add_left, add_left]
