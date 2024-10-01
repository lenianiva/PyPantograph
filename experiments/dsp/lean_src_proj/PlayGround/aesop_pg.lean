import Aesop

example : α → α :=
  by aesop

theorem eg2 : α → α := by
  intro a
  exact a
