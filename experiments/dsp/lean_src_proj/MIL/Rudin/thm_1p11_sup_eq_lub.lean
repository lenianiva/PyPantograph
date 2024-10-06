/-
1.11 Theorem Suppose $S$ is an ordered set with the least-upper-bound property, $B \subset S, B$ is not empty, and $B$ is bounded below.
Let $L$ be the set of all lower bounds of $B$. Then
exists in $S$, and $\alpha=\inf B$.
$$
\alpha=\sup L
$$
In particular, inf $B$ exists in $S$.

Proof
Since $B$ is bounded below, $L$ is not empty. Since $L$ consists of exactly those $y \in S$ which satisfy the inequality $y \leq x$ for every $x \in B$, we see that every $x \in B$ is an upper bound of $L$. Thus $L$ is bounded above. Our hypothesis about $S$ implies therefore that $L$ has a supremum in $S$; call it $\alpha$.

If $\gamma<\alpha$ then (see Definition 1.8) $\gamma$ is not an upper bound of $L$, hence $\gamma \notin B$. It follows that $\alpha \leq x$ for every $x \in B$. Thus $\alpha \in L$.

If $\alpha<\beta$ then $\beta \notin L$, since $\alpha$ is an upper bound of $L$.

We have shown that $\alpha \in L$ but $\beta \notin L$ if $\beta>\alpha$. In other words, $\alpha$ is a lower bound of $B$, but $\beta$ is not if $\beta>\alpha$. This means that $\alpha=\inf B$.
QED
-/

-- p1: α = sup L exists in S (trivial because it's in the way α is defined)
-- p2: inf B exist in S (trivial similar reasons)
-- p3: show α = inf B

-- S is a total order
-- B is not empty, B subset of S
-- L lower bounds of B
-- α = sup L
-- --> WTS: exists in α  S
import Mathlib.Init.Order.Defs
import Mathlib.Init.Set
import Mathlib.Order.Bounds.Basic

#check lowerBounds

-- https://chat.openai.com/c/45037233-fdad-47f2-879a-2f03eed09d02 (explanation of thm stmt)
theorem sup_of_lower_bounds {S : Type*} [LinearOrder S]
    (B : Set S) (L : Set S) (hL : L = lowerBounds B) (hL : L.Nonempty)
    {α : S} (hα : IsLUB L α) :
    IsGLB B α := by
  subst L
  simp only [IsGLB, IsLUB, IsLeast, IsGreatest] at *
  constructor
  · have := lowerBounds_mono_set (subset_upperBounds_lowerBounds B)
    exact this hα.2
  · exact hα.1
