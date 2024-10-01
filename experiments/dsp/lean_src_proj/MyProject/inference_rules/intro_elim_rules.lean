/-
1. Understand introduction rules
-> can they be applied to the the hypothesis?
-/

variable (p q : Prop)

#check And.intro -- And.intro {a b : Prop} (left : a) (right : b) : a ∧ b
example (hp : p) (hq : q) : p ∧ q := And.intro hp hq
example (hp : p) (hq : q) : p ∧ q := by
  apply And.intro
  exact hp
  exact hq

#check fun (hp : p) (hq : q) => And.intro hp hq

-- and elim
#check And.left -- And.left {a b : Prop} (h : a ∧ b) : a

variable (p q : Prop)

example (h : p ∧ q) : p := And.left h
example (h : p ∧ q) : q := And.right h

example (h : p ∧ q) : p := by
  apply And.left
  exact h
