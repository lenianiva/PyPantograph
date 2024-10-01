-- https://proofassistants.stackexchange.com/questions/4000/how-to-write-a-proof-of-n-0-0-in-lean-4-using-the-built-in-definition-of-add
import Mathlib.Data.Nat.Basic
import Aesop

-- ------------------------------
-- Prove that n + 0 = n
theorem n_plus_zero_normal : ∀ n : ℕ, n + 0 = n := by
  -- Consider some n in Nats.\n
  intro n
  -- Using facts of addition simplify n + 0 to n.
  rw [Nat.add_zero]

-- Prove that n + 0 = n via a formal proof sketch
theorem n_plus_zero_proved_formal_sketch : ∀ n : ℕ, n + 0 = n := by
  -- We have the fact of addition n + 0 = n, use it to show left and right are equal.
  have h_nat_add_zero: ∀ n : ℕ, n + 0 = n := Nat.add_zero
  exact h_nat_add_zero

-- Prove that n + 0 = n via a formal proof sketch with aesop
theorem n_plus_zero_proved_formal_sketch' : ∀ n : ℕ, n + 0 = n := by
  -- We have the fact of addition n + 0 = n, use it to show left and right are equal.
  have h_nat_add_zero: ∀ n : ℕ, n + 0 = n := by aesop
  exact h_nat_add_zero
-- ------------------------------

-- ------------------------------
-- Prove that 0 + n = n by induction
theorem zero_plus_n_normal : ∀ n : ℕ, 0 + n = n := by
  -- Consider some n in Nats.
  intro n
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: 0 + 0 = 0
    rw [Nat.add_zero]
  | succ n ih =>
    -- Inductive step: assume 0 + n = n, prove 0 + succ n = succ n
    rw [Nat.add_succ]
    rw [ih]

-- Prove that 0 + n = n by induction via a formal proof sketch
theorem zero_plus_n_proved_formal_sketch : ∀ n : ℕ, 0 + n = n := by
  -- Consider some n in Nats.
  intro n
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: 0 + 0 = 0
    have h_base: 0 + 0 = 0 := by rw [Nat.add_zero]
    exact h_base
  | succ n ih =>
    -- Inductive step: assume 0 + n = n, prove 0 + succ n = succ n
    have h_inductive: 0 + Nat.succ n = Nat.succ n := by
      rw [Nat.add_succ]
      rw [ih]
    exact h_inductive

-- Prove that 0 + n = n by induction via a formal proof sketch with aesop
theorem zero_plus_n_proved_formal_sketch' : ∀ n : ℕ, 0 + n = n := by
  -- Consider some n in Nats.
  intro n
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: 0 + 0 = 0
    have h_base: 0 + 0 = 0 := by aesop
    exact h_base
  | succ n ih =>
    -- Inductive step: assume 0 + n = n, prove 0 + succ n = succ n
    have h_inductive: 0 + Nat.succ n = Nat.succ n := by aesop
    exact h_inductive
-- ------------------------------

-- ------------------------------
-- darn, should've follwed sf here: https://github.com/marshall-lee/software_foundations/blob/master/lf/Induction.v#L207
-- Prove that n + (m + 1) = (n + m) + 1
theorem plus_n_Sm_normal : ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := by
  intros n m
  rw [Nat.add_succ]

-- Prove that n + (m + 1) = (n + m) + 1 via a formal proof sketch
theorem plus_n_Sm_proved_formal_sketch : ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := by
  -- We have then n + (m + 1) = (n + m) + 1 by the definition of addition.
  have h_nat_add_succ: ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := Nat.add_succ
  exact h_nat_add_succ

-- Prove that n + (m + 1) = (n + m) + 1 via a formal proof sketch with aesop
theorem plus_n_Sm_proved_formal_sketch' : ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := by
  -- We have then n + (m + 1) = (n + m) + 1 by the definition of addition.
  have h_nat_add_succ: ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := by aesop
  exact h_nat_add_succ
-- ------------------------------

-- ------------------------------
-- Prove that n + m = m + n
theorem add_comm_normal : ∀ n m : ℕ, n + m = m + n := by
  -- Consider some n and m in Nats.
  intros n m
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + m = m + 0.
    -- Using the definition of addition, 0 + m = m and m + 0 = m.
    rw [Nat.zero_add, Nat.add_zero]
  | succ n ih =>
    -- Inductive step: Assume n + m = m + n, we need to show succ n + m = m + succ n.
    have plus_n_Sm_normal: ∀ n m : ℕ, n + (m + 1) = (n + m) + 1 := by intros n m; rw [Nat.add_succ]
    rw [plus_n_Sm_normal, Nat.add_succ, Nat.add_zero]
    rw [← ih]
    rw [Nat.succ_add]

-- Prove that n + m = m + n via a formal proof sketch
theorem add_comm_proved_formal_sketch : ∀ n m : ℕ, n + m = m + n := by
  -- Consider some n and m in Nats.
  intros n m
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + m = m + 0.
    -- We have the fact 0 + m = m by the definition of addition.
    have h_base: 0 + m = m := Nat.zero_add m
    -- We also have the fact m + 0 = m by the definition of addition.
    have h_symm: m + 0 = m := Nat.add_zero m
    -- Combining these, we get 0 + m = m + 0.
    rw [h_base, h_symm]
  | succ n ih =>
    -- Inductive step: Assume n + m = m + n, we need to show succ n + m = m + succ n.
    -- By the inductive hypothesis, we have n + m = m + n.
    have h_inductive: n + m = m + n := ih
    -- proof is:
    -- We eventually want to flip n + m and simplifying to make both sides the same. Thus,
    -- 1. Note we start with: Nat.succ n + m = m + Nat.succ n, so, pull the succ out from m + Nat.succ n on the right side from the addition using addition facts Nat.add_succ.
    have h_pull_succ_out_from_right: m + Nat.succ n = Nat.succ (m + n) := by rw [Nat.add_succ]
    -- 2. then to flip m + S n to something like S (n + m) we need to use the IH.
    have h_flip_n_plus_m: Nat.succ (n + m) = Nat.succ (m + n) := by rw [h_inductive]
    -- 3. Now the n & m are on the correct sides Nat.succ n + m = Nat.succ (n + m), so let's use the def of addition to pull out the succ from the addition on the left using Nat.succ_add.
    have h_pull_succ_out_from_left: Nat.succ n + m = Nat.succ (n + m) := by rw [Nat.succ_add]
    -- Combining these, we get succ n + m = m + succ n.
    rw [h_pull_succ_out_from_right, ←h_flip_n_plus_m, h_pull_succ_out_from_left]

-- Prove that n + m = m + n via a formal proof sketch
theorem add_comm_proved_formal_sketch_aesop : ∀ n m : ℕ, n + m = m + n := by
  -- Consider some n and m in Nats.
  intros n m
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + m = m + 0.
    -- We have the fact 0 + m = m by the definition of addition.
    have h_base: 0 + m = m := Nat.zero_add m
    -- We also have the fact m + 0 = m by the definition of addition.
    have h_symm: m + 0 = m := Nat.add_zero m
    -- Combining these, we get 0 + m = m + 0.
    rw [h_base, h_symm]
  | succ n ih =>
    -- Inductive step: Assume n + m = m + n, we need to show succ n + m = m + succ n.
    -- By the inductive hypothesis, we have n + m = m + n.
    have h_inductive: n + m = m + n := ih
    -- proof is:
    -- We eventually want to flip n + m and simplifying to make both sides the same. Thus,
    -- 1. Note we start with: Nat.succ n + m = m + Nat.succ n, so, pull the succ out from m + Nat.succ n on the right side from the addition using addition facts Nat.add_succ.
    have h_pull_succ_out_from_right: m + Nat.succ n = Nat.succ (m + n) := by aesop
    -- 2. then to flip m + S n to something like S (n + m) we need to use the IH.
    have h_flip_n_plus_m: Nat.succ (n + m) = Nat.succ (m + n) := by aesop
    -- 3. Now the n & m are on the correct sides Nat.succ n + m = Nat.succ (n + m), so let's use the def of addition to pull out the succ from the addition on the left using Nat.succ_add.
    have h_pull_succ_out_from_left: Nat.succ n + m = Nat.succ (n + m) := by rw [Nat.succ_add]
    -- Combining these, we get succ n + m = m + succ n.
    rw [h_pull_succ_out_from_right, ←h_flip_n_plus_m, h_pull_succ_out_from_left]
-- ------------------------------

-- ------------------------------
-- Prove that n + (m + p) = (n + m) + p
theorem add_assoc_normal : ∀ n m p : ℕ, n + (m + p) = (n + m) + p := by
  -- Consider some n, m, and p in Nats.
  intros n m p
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + (m + p) = (0 + m) + p.
    -- Using the definition of addition, 0 + (m + p) = m + p and (0 + m) + p = m + p.
    rw [Nat.zero_add, Nat.zero_add]
  | succ n ih =>
    -- Inductive step: Assume n + (m + p) = (n + m) + p, we need to show succ n + (m + p) = (succ n + m) + p.
    -- proof strategy is, we move succ n out (or in) enough times then use the IH until both sides are the same.
    -- 1. let's start by pulling out the scc from left side and have the entire addition inside the succ.
    rw [Nat.succ_add]
    -- 2. Now that we have the IH hypothesis appearing inside the left, let's apply it so we have n + (m + p) = (n + m) + p.
    rw [ih]
    -- 3. Now that the parenthesis (apps of plus) are on the right place for both side, push the succ on the left twice so both terms are the same.
    rw [← Nat.succ_add, ← Nat.succ_add]

-- Prove that n + (m + p) = (n + m) + p
theorem add_assoc_proved_formal_sketch : ∀ n m p : ℕ, n + (m + p) = (n + m) + p := by
  -- Consider some n, m, and p in Nats.
  intros n m p
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + (m + p) = (0 + m) + p.
    -- Using the definition of addition, 0 + (m + p) = m + p and (0 + m) + p = m + p.
    rw [Nat.zero_add, Nat.zero_add]
  | succ n ih =>
    -- Inductive step: Assume n + (m + p) = (n + m) + p, we need to show succ n + (m + p) = (succ n + m) + p.
    -- proof strategy is, we move succ n out (or in) enough times then use the IH until both sides are the same.
    -- 1. let's start by pulling out the scc from left side and have the entire addition inside the succ.
    have h_pull_add_succ_out_from_left: Nat.succ n + (m + p) = Nat.succ (n + (m + p)) := by rw [Nat.succ_add]
    -- 2. Now that we have the IH hypothesis appearing inside the left, let's apply it so we have n + (m + p) = (n + m) + p.
    have h_inside_left_associates: Nat.succ (n + (m + p)) = Nat.succ ((n + m) + p) := by rw [ih]
    -- 3. Now that the parenthesis (apps of plus) are on the right place for both side, push the succ on the left twice so both terms are the same.
    have h_push_succ_in_left_twice: Nat.succ ((n + m) + p) = ((Nat.succ n) + m) + p := by rw [← Nat.succ_add, ← Nat.succ_add]
    -- Combining these, we get succ n + (m + p) = (succ n + m) + p.
    rw [h_pull_add_succ_out_from_left, h_inside_left_associates, h_push_succ_in_left_twice]

-- Prove that n + (m + p) = (n + m) + p via a formal proof sketch with aesop
theorem add_assoc_proved_formal_sketch_aesop : ∀ n m p : ℕ, n + (m + p) = (n + m) + p := by
  -- Consider some n, m, and p in Nats.
  intros n m p
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + (m + p) = (0 + m) + p.
    -- Using the definition of addition, 0 + (m + p) = m + p and (0 + m) + p = m + p.
    rw [Nat.zero_add, Nat.zero_add]
  | succ n ih =>
    -- Inductive step: Assume n + (m + p) = (n + m) + p, we need to show succ n + (m + p) = (succ n + m) + p.
    -- proof strategy is, we move succ n out (or in) enough times then use the IH until both sides are the same.
    -- 1. let's start by pulling out the scc from left side and have the entire addition inside the succ.
    have h_pull_add_succ_out_from_left: Nat.succ n + (m + p) = Nat.succ (n + (m + p)) := by rw [Nat.succ_add]
    -- 2. Now that we have the IH hypothesis appearing inside the left, let's apply it so we have n + (m + p) = (n + m) + p.
    have h_inside_left_associates: Nat.succ (n + (m + p)) = Nat.succ ((n + m) + p) := by aesop
    -- 3. Now that the parenthesis (apps of plus) are on the right place for both side, push the succ on the left twice so both terms are the same.
    have h_push_succ_in_left_twice: Nat.succ ((n + m) + p) = ((Nat.succ n) + m) + p := by rw [← Nat.succ_add, ← Nat.succ_add]
    -- Combining these, we get succ n + (m + p) = (succ n + m) + p.
    rw [h_pull_add_succ_out_from_left, h_inside_left_associates, h_push_succ_in_left_twice]
-- ------------------------------
