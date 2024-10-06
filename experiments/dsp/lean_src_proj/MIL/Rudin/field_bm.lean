/-
\section{FIELDS}

1.12 Definition A field is a set $F$ with two operations,
called addition and multiplication, which satisfy the
following so-called "field axioms" (A), (M), and (D):

\section{(A) Axioms for addition}

(A1) If $x \in F$ and $y \in F$, then their sum $x+y$ is in $F$.

(A2) Addition is commutative: $x+y=y+x$ for all $x, y \in F$.

(A3) Addition is associative: $(x+y)+z=x+(y+z)$ for all $x, y, z \in F$.

(A4) $F$ contains an element 0 such that $0+x=x$ for every $x \in F$.

(A5) To every $x \in F$ corresponds an element $-x \in F$ such that

$$
x+(-x)=0
$$

(M) Axioms for multiplication

(M1) If $x \in F$ and $y \in F$, then their product $x y$ is in $F$.

(M2) Multiplication is commutative: $x y=y x$ for all $x, y \in F$.

(M3) Multiplication is associative: $(x y) z=x(y z)$ for all $x, y, z \in F$.

(M4) $F$ contains an element $1 \neq 0$ such that $1 x=x$ for every $x \in F$.

(M5) If $x \in F$ and $x \neq 0$ then there exists an element $1 / x \in F$ such that

$$
x \cdot(1 / x)=1 \text {. }
$$

\section{(D) The distributive law}

$$
x(y+z)=x y+x z
$$

holds for all $x, y, z \in F$.

https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html#
-/

class Field (α : Type) where
  add : α → α → α
  comm: ∀ x y : α, add x y = add y x
  ass: ∀ x y z : α, (add (add x y) z) = (add x (add x z))
  mul : α → α → α

#check Field.mk
#check Field Nat

-- (A1) If $x \in F$ and $y \in F$, then their sum $x+y$ is in $F$.
-- closure is not needed since add implies

-- (A2) Addition is commutative: $x+y=y+x$ for all $x, y \in F$.
-- since we are defining what the structure (class) is, then we define that it is commutative (then later you can show specifc objects like Q, Reals are Fields)

-- Goal prove rationals are a field (to debug my definition of field) & use def of rationals

-- structure == inductive, but structure has one constructor
-- class ≈ structure + other machinery
/-
structure foo_s where
  x : Nat
  y : Nat
#check foo_s.mk
inductive foo_i
| mk (x y : Nat)

def foo_i.x : foo_i → Nat
| mk x _ => x

#check foo_s.x
#check foo_i.x
-/
