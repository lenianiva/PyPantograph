/-

{
    "problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
    "level": "Level 3",
    "type": "Algebra",
    "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes."
}

theorem:the graph of y=2/(x^2+x-6) has 2 vertical asymptotes.
Proof.
Define vertical asymptote as lim_{x->c} f(x) = ∞ or -∞.
The denominator of the rational function factors into x^2+x-6=(x-2)(x+3).
Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is 0,
which occurs for x = 2 and x = -3.
Therefore, the graph has 2 vertical asymptotes.
Qed.
-/

import Mathlib.Data.Real.Basic

-- noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 + x - 6)
noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 + x - 6)
#check f
