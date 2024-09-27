https://www.evernote.com/shard/s410/nl/75276202/2170cbbd-24a1-2d25-da32-bd8f3270d190?title=prompt%20for%20creating%20toy%20example
https://chatgpt.com/c/0ad32608-cbc9-4627-a705-786ed7421826
I want all final responses in this format:
```json
{
        "nl_problem": ["Prove that for any natural number n, n + 0 = n."],
        "nl_solution": [
            "Consider any natural number n.",
            "Using the properties of addition, we know that adding zero to any number does not change the value of that number.",
            "Therefore, we can conclude that n + 0 = n."
        ],
        "nl_solution_sketch": [
            "Consider any natural number n.",
            "From properties of addition, adding zero does not change its values.",
            "Thus, n + 0 = n."
        ],
        "fl_problem": ["theorem n_plus_zero_normal : ∀ n : ℕ, n + 0 = n :="],
        "fl_partial_sketch": [
            "-- Prove that n + 0 = n via a formal proof sketch with holes to be filled\n",
            "theorem n_plus_zero_proved_formal_sketch'' : ∀ n : ℕ, n + 0 = n := by\n",
            "  -- We have the fact of addition n + 0 = n, use it to show left and right are equal.\n",
            "  have h_nat_add_zero: ∀ n : ℕ, n + 0 = n := <TODO_PROOF_OR_HAMMER>\n",
            "  -- Combine facts with to close goal\n",
            "  <TODO_PROOF_OR_HAMMER>\n"
        ],
        "src_header_fl_problem": ["import Mathlib.Data.Nat.Basic"], 
        "fl_header_sketch":  [
            "import Mathlib.Data.Nat.Basic",
            "import Aesop"
        ],
        "path_2_file": "~/gold-ai-olympiad/lean_src_proj/lean_basics/basic_nats_using_mathlib_nats2_simp_no_rw.lean",
        "fl_statement_idx": "0"
    },



{
        "nl_problem": ["Prove that for any natural number n, 0 + n = n."],
        "nl_solution": [
            "Consider any natural number n. We will prove the statement by induction on n.",
            "Base case: When n = 0, we need to show that 0 + 0 = 0. This is true by the definition of addition.",
            "Inductive step: Assume that for some natural number n, 0 + n = n. We need to show that 0 + (n + 1) = (n + 1). By the definition of addition and the inductive hypothesis, we have 0 + (n + 1) = (0 + n) + 1 = n + 1. Therefore, the statement holds for n + 1.",
            "Thus, by induction, we have proved that for any natural number n, 0 + n = n."
        ],
        "nl_solution_sketch": [
            "Consider any natural number n, and do induction on n.",
            "Base case: 0 + 0 = 0 by properties of addition.",
            "Inductive step we have 0 + n = n. Then 0 + (n + 1) = (0 + n) + 1 = n + 1.",
            "Where, 0 + n = n by assumption,qed."
        ],
        "fl_problem": ["theorem zero_plus_n_proved_formal_sketch : ∀ n : ℕ, 0 + n = n :="],
        "fl_partial_sketch": [
            "-- Prove that 0 + n = n by induction via a formal proof sketch with holes to be filled\n",
            "theorem zero_plus_n_proved_formal_sketch'' : ∀ n : ℕ, 0 + n = n := by\n",
            "  -- Consider some n in Nats.\n",
            "  intro n\n",
            "  -- Perform induction on n.\n",
            "  induction n with\n",
            "  | zero =>\n",
            "    -- Base case: 0 + 0 = 0\n",
            "    have h_base: 0 + 0 = 0 := <TODO_PROOF_OR_HAMMER>\n",
            "    -- Combine facts to close goal\n",
            "    <TODO_PROOF_OR_HAMMER>\n",
            "  | succ n ih =>\n",
            "    -- Inductive step: assume 0 + n = n, prove 0 + succ n = succ n\n",
            "    have h_inductive: 0 + Nat.succ n = Nat.succ n := <TODO_PROOF_OR_HAMMER>\\n",
            "    -- Combine facts to close goal\n",
            "    <TODO_PROOF_OR_HAMMER>\n"
        ],
        "src_header_fl_problem": ["import Mathlib.Data.Nat.Basic"], 
        "fl_header_sketch":  [
            "import Mathlib.Data.Nat.Basic",
            "import Aesop"
        ]
    }
```
I want to translate the following formal proof (solution) in lean 4 to a natural language proof (solution) that a human would write (without lean code in it) and eventually make it into a concise nl_solution_sketch, like the following one:
```human_problem_solution_proof.json
"nl_problem": ["Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."],
"nl_solution_sketch": ["For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."]
``` 
This is my lean 4 fl theorem (fl problem) and fl proof (fl solution):
```
-- Prove that n + (m + p) = (n + m) + p
theorem add_assoc_proved_formal_sketch : ∀ n m p : ℕ, n + (m + p) = (n + m) + p := by
  -- Consider some n, m, and p in Nats.
  intros n m p
  -- Perform induction on n.
  induction n with
  | zero =>
    -- Base case: When n = 0, we need to show 0 + (m + p) = (0 + m) + p.
    -- Using the definition of addition, 0 + (m + p) = m + p and (0 + m) + p = m + p.
    simp [Nat.zero_add, Nat.zero_add]
  | succ n ih =>
    -- Inductive step: Assume n + (m + p) = (n + m) + p, we need to show succ n + (m + p) = (succ n + m) + p.
    -- proof strategy is, we move succ n out (or in) enough times then use the IH until both sides are the same.
    -- 1. let's start by pulling out the scc from left side and have the entire addition inside the succ.
    have h_pull_add_succ_out_from_left: Nat.succ n + (m + p) = Nat.succ (n + (m + p)) := by simp [Nat.succ_add]
    -- 2. Now that we have the IH hypothesis appearing inside the left, let's apply it so we have n + (m + p) = (n + m) + p.
    have h_inside_left_associates: Nat.succ (n + (m + p)) = Nat.succ ((n + m) + p) := by simp [ih]
    -- 3. Now that the parenthesis (apps of plus) are on the right place for both side, push the succ on the left twice so both terms are the same.
    have h_push_succ_in_left_twice: Nat.succ ((n + m) + p) = ((Nat.succ n) + m) + p := by simp [Nat.succ_add, Nat.succ_add]
    -- Combining these, we get succ n + (m + p) = (succ n + m) + p.
    simp [h_pull_add_succ_out_from_left, h_inside_left_associates, h_push_succ_in_left_twice]
``` 
use the comments to translate the fl proof (solution) to natural language solution then use that to output a natural language concise sketch. Make the natural language proof (solution) sketch concise with the core elements of the solution proof. Do this by first outputting the natural language solution, distill it into a very concise proof sketch in natural language with only the core components. Output everything in a json code block please: