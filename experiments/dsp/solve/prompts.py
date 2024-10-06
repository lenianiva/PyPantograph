"""
core part of data for prompt for dsp:
"nl_problem": ..., # x*_nl
"nl_solution": ..., # y*_nl = draft*
"fl_problem": ...,  # x*_fl
"fl_partial_sketch": ...,  # z_fl example = sketch
"src_header_fl_problem": ..., #src_header_x*_fl
"fl_header_sketch": ...,  # hz_fl suggested header
"""
import json, sys, unittest
from pathlib import Path
from typing import Optional

experiment_dir = Path(__file__).resolve().parent.parent

# just an example of stop tokens from the MATH eval code
# STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

default_path_2_examples = 'debug/toy_example1_dsp/dsp_debug5_sf/dsp_debug5_sf_train.json'

TOKEN_PLACEHOLDER = "<TODO_PROOF_OR_HAMMER>"

# -- Prompt draft (P_draft) for Lean 4
"""
Draft an informal solution similar to the one below. 
The informal solution will be used to sketch a formal proof in the Lean 4 Proof Assistant.
Here are some examples: 

Informal: 
(*### Problem\n\n 
[...nl/i problem text...]\n\n
### Solution\n\n
[...nl/i solution/draft text...]\n\n
*)\n\n

Informal:
(*### Problem\n\n
{nl_problem}
### Solution\n\n
[...Model Completion...]
"""
SYSTEM_PROMPT_DRAFT_V0 = 'You are an expert mathematician and an expert in the Lean 4 Proof Assistant.'
STOP_TOKENS_DRAFT_V0: list[str] = ['Informal:', '(*### Problem']
prompt_draft_template_lean4_v0 = ("Draft an informal solution similar to the one below. "
"The informal solution will be used to sketch a formal proof in the Lean 4 Proof Assistant. "
"Here are some examples of informal problem solutions pairs:\n")
def get_prompt_draft_template_4_lean_v0(
    path_2_examples: str = default_path_2_examples,
    start: int = 0, 
    end: int = sys.maxsize,
    prompt_draft_template_4_lean: Optional[str] = prompt_draft_template_lean4_v0, 
    verbose: bool = False,
    ):
    path_2_examples = experiment_dir / Path(path_2_examples)
    # load json file with list of dicts from file in one line
    with open(path_2_examples, 'r') as f:
        examples: list[dict] = json.load(f)
    print(f'{len(examples)=}') if verbose else None
    examples = examples[start:end]
    # -- Create prompt by appending few shot examples
    for example in examples:
        nl_problem = example['nl_problem']
        new_few_shot_example = "\nInformal:\n(*### Problem\n\n" + ' '.join(nl_problem)
        nl_solution_sketch = example['nl_solution_sketch']
        new_few_shot_example += "\n\n### Solution\n\n" + ' '.join(nl_solution_sketch) + "*)\n"
        prompt_draft_template_4_lean += new_few_shot_example
    # Add part to elicit model to do task
    prompt_draft_template_4_lean += "\nInformal: \n(*### Problem\n\n{nl_problem}\n\n### Solution\n"
    # Return
    print(prompt_draft_template_4_lean) if verbose else None
    return prompt_draft_template_4_lean
prompt_draft_template_lean4_v0 = get_prompt_draft_template_4_lean_v0() 

# -- Prompt sketch (P_sketch) for Lean 4
"""
[... Translate informal draft to a formal sketch in Lean 4. Here are some examples: ...]
Informal:\n
(*### Problem\n\n
[...nl/i problem text...]\n\n
### Solution\n\n
[...nl/i solution/draft text...]\n\n
*)\n\n
Formal:\n
[...fl/i problem text...]
[...fl/i partial sketch text...]
\n\n

Informal:\n
(*### Problem\n\n
{nl_problem}
### Solution\n\n
{nl_solution}
*)\n\n
Formal:\n
{fl_problem}
[...Model Completion...]
"""
# tasks is mostly writing lean but perhaps making it think it's good at maths is also good? we could later test just focusing system prompting it to be good at Lean 4. 
SYSTEM_PROMPT_SKETCH_V0 = 'You are an expert mathematician and an expert in the Lean 4 Proof Assistant.'
STOP_TOKENS_SKETCH_V0: list[str] = ['Informal:', '(*### Problem', '###Solution', 'Formal:']
prompt_sketch_template_lean4_v0 = ("Translate the informal solution into a sketch in the "
f"formal Lean 4 proof. Add {TOKEN_PLACEHOLDER} in the formal sketch whenever possible. "
f"{TOKEN_PLACEHOLDER} will be used to call a automated theorem prover or tactic in Lean 4. Do not use any lemmas."
"Here are some examples:\n"
)
def get_prompt_sketch_template_4_lean_v0(
    path_2_examples: str = default_path_2_examples,
    start: int = 0, 
    end: int = sys.maxsize,
    prompt_sketch_template_4_lean: Optional[str] = prompt_sketch_template_lean4_v0, 
    autoformalize_prob_in_prompt: Optional[bool] = False,
    verbose: bool = False,
    ):
    path_2_examples = experiment_dir / Path(path_2_examples)
    # load json file with list of dicts from file in one line
    with open(path_2_examples, 'r') as f:
        examples: list[dict] = json.load(f)
    print(f'{len(examples)=}') if verbose else None
    examples = examples[start:end]
    # -- Create prompt by appending few shot examples
    for example in examples:
        # TODO: might need to figure out the header thing
        nl_problem = example['nl_problem']
        new_few_shot_example = "\nInformal:\n(*### Problem\n\n" + ' '.join(nl_problem)
        nl_solution_sketch = example['nl_solution_sketch']
        new_few_shot_example += "\n\n### Solution\n\n" + ' '.join(nl_solution_sketch) + "*)\n"
        fl_problem = example['fl_problem']
        fl_header_sketch = example['fl_header_sketch']
        fl_header_sketch = '\n'.join(fl_header_sketch) + '\n\n'
        new_few_shot_example += "\nFormal:\n"+ fl_header_sketch + ' '.join(fl_problem)
        fl_partial_sketch = example['fl_partial_sketch']
        new_few_shot_example += ' '.join(fl_partial_sketch)
        prompt_sketch_template_4_lean += new_few_shot_example
    # Add part to elicit model to do task
    if autoformalize_prob_in_prompt:
        prompt_sketch_template_4_lean += "\nInformal:\n(*### Problem\n\n{nl_problem}\n\n### Solution\n\n{nl_solution}*)\n\nFormal:\n"
    else:
        prompt_sketch_template_4_lean += "\nInformal:\n(*### Problem\n\n{nl_problem}\n\n### Solution\n\n{nl_solution}*)\n\nFormal:\n{fl_problem}"
    # Return
    print(prompt_sketch_template_4_lean) if verbose else None
    return prompt_sketch_template_4_lean
prompt_sketch_template_lean4_v0 = get_prompt_sketch_template_4_lean_v0()

WALL = "```"


def extract_lean_code(
        sketch: str,
        placeholder: str = TOKEN_PLACEHOLDER,
        strip_imports: bool = True) -> list[str]:
    lines = sketch.split("\n")
    # find backtick markers ```
    lean_codes = []
    curr = []
    is_walled = False
    is_walled_lean = False
    for line in lines:
        if not is_walled:
            if line.rstrip() == f"{WALL}lean":
                is_walled = True
                is_walled_lean = True
            elif line.startswith(WALL):
                is_walled = True
                is_walled_lean = False
            continue

        if line.rstrip() == WALL:
            if is_walled_lean:
                # found wall
                code = "\n".join(curr) + "\n"
                code = code.replace("ℕ", "Nat").replace(placeholder, "sorry")
                lean_codes.append(code)
                curr = []
            is_walled = False
            is_walled_lean = False
            continue

        if strip_imports and line.startswith("import "):
            continue
        curr.append(line)

    return lean_codes


class TestPrompts(unittest.TestCase):

    def test_extract_lean_code(self):
        sketch = "```lean\nimport Mathlib.Data.Nat.Basic\nimport Aesop\n\ntheorem n_plus_zero : ∀ n : ℕ, n + 0 = n := by\n   -- Consider any natural number n. We need to show that n + 0 = n.\n   -- Use the fact that adding zero to any natural number does not change its value.\n   have h_nat_add_zero: ∀ n : ℕ, n + 0 = n := <TODO_PROOF_OR_HAMMER>\n   -- Combine facts to close goal\n   <TODO_PROOF_OR_HAMMER>\n```"
        codes = extract_lean_code(sketch)
        self.assertEqual(codes, [
            "\ntheorem n_plus_zero : ∀ n : Nat, n + 0 = n := by\n   -- Consider any natural number n. We need to show that n + 0 = n.\n   -- Use the fact that adding zero to any natural number does not change its value.\n   have h_nat_add_zero: ∀ n : Nat, n + 0 = n := sorry\n   -- Combine facts to close goal\n   sorry\n"
        ])

    def test_extract_sketch_minif2f(self):
        sketch = "To solve the problem formally in Lean 4, we will sketch the proof by breaking down the steps of the informal solution, leveraging Lean 4 tactics and possibly automated theorem proving. Here's how the formal sketch might look:\n\n```lean\nimport Mathlib.Data.Complex.Basic\nimport Aesop\n\n-- Define the complex number z\ndef z : ℂ := (1 + Complex.i) / Real.sqrt 2\n\ntheorem complex_problem_solution : \n  (∑ i in finset.range 12, z ^ (i + 1) ^ 2) * \n  (∑ i in finset.range 12, z ^ -((i + 1) ^ 2)) = 36 := by\n  -- We first compute z as a complex number with a modulus of 1.\n  -- Thus, powers of z represent rotations in the complex plane.\n  have h_mod_z : Complex.abs z = 1 := <TODO_PROOF_OR_HAMMER>\n  -- Recognize that each term z^(k^2) and its reciprocal are conjugates due to modulus 1.\n  have h_conjugates : ∀ k : ℕ, z^(k^2) * (z^(-k^2)) = 1 := <TODO_PROOF_OR_HAMMER>\n  -- The product (z^(1^2) + z^(2^2) + ... + z^(12^2)) * (z^(-1^2) + z^(-2^2) + ... + z^(-12^2))\n  -- simplifies as a result of this conjugate pair property.\n  have h_sum_conjugates : ∑ i in finset.range 12, z ^ (i + 1) ^ 2 = 0 := <TODO_PROOF_OR_HAMMER>\n  have h_sum_reciprocals : ∑ i in finset.range 12, z ^ -((i + 1) ^ 2) = 0 := <TODO_PROOF_OR_HAMMER>\n  -- Combine all the contributions, where non-zero terms contribute to the total sum\n  -- to ensure the final product is 36 based on the angle and distribution of terms.\n  have h_final_sum_product : (∑ i in finset.range 12, z ^ (i + 1) ^ 2) *\n                             (∑ i in finset.range 12, z ^ -((i + 1) ^ 2)) = 36 := <TODO_PROOF_OR_HAMMER>\n  -- Conclude the proof with the calculated product.\n  exact h_final_sum_product\n```\n\nIn this sketch:\n- We define \\( z \\) as the complex number \\(\\frac{1+i}{\\sqrt{2}}\\).\n- We use properties of complex numbers with modulus 1, recognizing rotational symmetry and conjugate pair relations.\n- We use automated tactics (`<TODO_PROOF_OR_HAMMER>`) to handle calculations involving sums, products, and properties of the complex exponentials.\n- Finally, we show that the structure of these complex exponentials simplifies the product to yield the desired result of 36."
        codes = extract_lean_code(sketch)
        self.assertEqual(len(codes), 1)

if __name__ == '__main__':
    unittest.main()
