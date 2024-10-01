"""
core part of data for prompt for dsp:
"nl_problem": ..., # x*_nl
"nl_solution": ..., # y*_nl = draft*
"fl_problem": ...,  # x*_fl
"fl_partial_sketch": ...,  # z_fl example = sketch
"src_header_fl_problem": ..., #src_header_x*_fl
"fl_header_sketch": ...,  # hz_fl suggested header
"""
import json
import sys
from pathlib import Path
from typing import Optional

experiment_dir = Path(__file__).resolve().parent.parent

# just an example of stop tokens from the MATH eval code
# STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

default_path_2_examples = 'debug/toy_example1_dsp/dsp_debug5_sf/dsp_debug5_sf_train.json'

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
"formal Lean 4 proof. Add <TODO_PROOF_OR_HAMMER> in the formal sketch whenever possible. "
"<TODO_PROOF_OR_HAMMER> will be used to call a automated theorem prover or tactic in Lean 4. "
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

# -- Main

def main(
    verbose: bool = True,
):
    # -- Print Prompt Draft
    # print('-- Prompt Draft --')
    # print(prompt_draft_template_lean4_v0)

    # -- Print Prompt Sketch
    print('-- Prompt Sketch --')
    sketch_prompt: str = get_prompt_sketch_template_4_lean_v0(verbose=verbose)
    # print(prompt_sketch_template_lean4_v0)
    print(sketch_prompt)

if __name__ == '__main__':
    import time
    start = time.time()
    # fire.Fire()
    main()
    end = time.time()
    print(f'Time elapsed: {end - start} seconds, or {(end - start) / 60} minutes, or {(end - start) / 3600} hours.')
