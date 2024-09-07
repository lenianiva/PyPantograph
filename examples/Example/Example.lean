import Aesop

/-- Ensure that Aesop is running -/
example : α → α :=
  by aesop

example : ∀ (p q: Prop), p ∨ q → q ∨ p := by
  intro p q h
  -- Here are some comments
  cases h
  . apply Or.inr
    assumption
  . apply Or.inl
    assumption
