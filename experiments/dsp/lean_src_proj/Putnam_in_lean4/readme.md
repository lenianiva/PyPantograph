# Prompt
```text
Help me formalize the following problem into a Theorem in the Lean4 Proof Assistant:
<stmt_to_formalize>
For a positive integer n,
 let f_n(x) = cos(x) cos(2x) cos(3x)··· cos(nx).
 Find the smallest n such that | f''_n(0)| > 2023.
</stmt_to_formalize>
Formalize only the Theorem correctly, if we need using Mathlib let's use it.
The proof is not needed, only the theorem so write ":= sorry" for the proof.
```