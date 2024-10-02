import sys, os, json, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Any
from collections import namedtuple
from tqdm import tqdm
from openai import OpenAI
import wandb
from tenacity import retry, stop_after_attempt, wait_exponential
from pantograph import Server

from solve.prompts import (
    SYSTEM_PROMPT_DRAFT_V0,
    SYSTEM_PROMPT_SKETCH_V0,
    prompt_draft_template_lean4_v0,
    prompt_sketch_template_lean4_v0,
    STOP_TOKENS_DRAFT_V0,
    STOP_TOKENS_SKETCH_V0,
    get_prompt_sketch_template_4_lean_v0,
)

# prompt_draft_template_lean4_v0 = "Draft an informal solution similar to the one below. The informal solution will be used to sketch a formal proof in the Lean 4 Proof Assistant. Here are some examples of informal problem solutions pairs:\n\nInformal:\n(*### Problem\n\nProve that for any natural number n, n + 0 = n.\n\n### Solution\n\nConsider any natural number n. From properties of addition, adding zero does not change its values. Thus, n + 0 = n.*)\n\nInformal:\n(*### Problem\n\nProve that for any natural number n, n + (m + 1) = (n + m) + 1.\n\n### Solution\n\nConsider any natural numbers n and m. From properties of addition, adding 1 to the sum of n and m is the same as first adding m to n and then adding 1. Thus, n + (m + 1) = (n + m) + 1.*)\n\nInformal:\n(*### Problem\n\nProve that for any natural number n and m, n + m = m + n.\n\n### Solution\n\nConsider any natural numbers n and m. We will do induction on n. Base case: 0 + m = m + 0 by properties of addition. Inductive step, we have n + m = m + n. Then (n + 1) + m = (n + m) + 1 = (m + n) + 1 = m + (n + 1). Thus, by induction, n + m = m + n, qed.*)\n\nInformal: \n(*### Problem\n\n{nl_problem}\n\n### Solution\n"

@dataclass
class SamplingParams:
    n: int
    max_tokens: int
    top_p: int
    temperature: float
    stop: str

class Engine:
    def __init__(self):
        pass

    def __call__(self, *args, **kwards):
        pass

class OpenAI_DSP_Engine(Engine):
    def __init__(
                self,
                model: str,
                api_key: str = None,
                base_url: str = None,  # e.g., Mistral-7B-Instrcut-v0.2 on http://120.77.8.29:12345
                # Draft Params
                draft_system_prompt: str = SYSTEM_PROMPT_DRAFT_V0,  # 'You are an expert mathematician and an expert in the Lean 4 Proof Assistant.' (goal do draft)
                draft_prompt_template: str = prompt_draft_template_lean4_v0,
                draft_sampling_params = None,
                draft_stop_tokens: list[str] = STOP_TOKENS_DRAFT_V0,
                # Sketch Params
                sketch_system_prompt: str = SYSTEM_PROMPT_SKETCH_V0,
                sketch_prompt_template: str = prompt_sketch_template_lean4_v0,
                sketch_sampling_params = None,
                sketch_stop_tokens: list[str] = STOP_TOKENS_SKETCH_V0,
                # Prove Params
                # ...TODO not sure if needed right now...
                # Misc
                verbose_init: bool = True,
                ):
        super().__init__()
        print(f'{api_key=}, {base_url=}') if verbose_init else None


        if not ('gpt-4-' in model or 'gpt-3.5-' in model or 'gpt-4o' in model):
            raise ValueError(f"Model {model=} not supported.")
        self.model = model
        self.api_key = api_key
        self.llm = OpenAI(api_key=self.api_key, base_url=base_url)
        # Draft params
        self.draft_system_prompt = draft_system_prompt
        self.draft_prompt_template = draft_prompt_template
        self.draft_sampling_params = draft_sampling_params
        # self.draft_sampling_params.stop = draft_stop_tokens
        # Sketch params
        self.sketch_system_prompt = sketch_system_prompt
        self.sketch_prompt_template = sketch_prompt_template
        self.sketch_sampling_params = sketch_sampling_params
        # self.sketch_sampling_params.stop = sketch_stop_tokens
        # Prove params
        # ...TODO not sure if needed right now...

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def autoformalize_prob(
    eng,
    data_pt: dict,
    verbose: bool = False,
):
    """ Autoformalize natural language problem to formal language problem. """
    ...

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def draft(
    eng,
    data_pt: dict,
    verbose: bool = False,
    ) -> list:
    """
    Creates (informal nl) draft (nl soln, nl proof sketch) for latter use in a formal proof sketch.
        y_pred_nl ~ draft(eng, x_nl_prob, P_draft)
    """
    # Make prompt from template
    nl_problem: str = data_pt['nl_problem'][0]
    prompt = eng.draft_prompt_template.replace('{nl_problem}', nl_problem)
    # Get all **completions** to single prompt, one (in) -> many (out)
    # ref: https://platform.openai.com/docs/api-reference/chat/object
    response: Any = eng.llm.chat.completions.create(
        model=eng.model,
        messages=[
            {"role": "system", "content": eng.draft_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=eng.draft_sampling_params.temperature,
        top_p=eng.draft_sampling_params.top_p,
        n=eng.draft_sampling_params.n,
        stop=eng.draft_sampling_params.stop[:3],
        )
    # Get all completions for single prompt
    completions: list[str] = [completion.message.content for completion in response.choices]  # response.choices[i].message
    drafts: list[str] = completions
    return drafts

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def sketch(
    eng,
    data_pt: dict,
    drafts: list,
    autoformalize_prob_in_prompt: bool = False,
    verbose: bool = False,
    ) -> list:
    """
    Creates (formal fl) sketch (fl proof sketch) for latter use in a formal proof sketch.
        z_pred_fl ~ sketch(eng, x_nl_prob, y_pred_nl, x_fl_prob, P_sketch)
    """
    assert len(drafts) == 1, f"For now only 1 draft."
    # Make prompt from template
    x_nl_problem: str = data_pt['nl_problem'][0]
    y_nl_solution: str = drafts[0]
    x_fl_problem = None
    if autoformalize_prob_in_prompt:
        # prompt = eng.sketch_prompt_template.replace('{nl_problem}', x_nl_problem).replace('{nl_solution}', y_nl_solution)
        not NotImplemented
    else:
        x_fl_problem = data_pt['fl_problem'][0] if 'fl_problem' in data_pt else autoformalize_prob(eng, data_pt)
        prompt = eng.sketch_prompt_template.replace('{fl_problem}', x_nl_problem).replace('{fl_problem}', y_nl_solution)
    # Get all **completions** to single prompt, one (in) -> many (out), ref: https://platform.openai.com/docs/api-reference/chat/object
    response: Any = eng.llm.chat.completions.create(
        model=eng.model,
        messages=[
            {"role": "system", "content": eng.sketch_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=eng.sketch_sampling_params.temperature,
        top_p=eng.sketch_sampling_params.top_p,
        n=eng.sketch_sampling_params.n,
        # stop=eng.sketch_sampling_params.stop[:3],
    )
    # Get all completions for single prompt
    completions: list[str] = [completion.message.content for completion in response.choices]  # response.choices[i].message
    sketches: list[str] = completions
    # Return
    return sketches, x_fl_problem

def prove(
    eng,
    fl_prob: str,
    fl_sketch: list[str],
):
    """

    Complete formal sketch and check if it proves the theorem.

    fl_prob --> Lean4 theorem (problem)
    fl_sketch --> Lean4 Form Sketch --> have x have ha

    """
    print(f"fl_prob={fl_prob}")
    print(f"fl_sketch={fl_sketch}")
    raise RuntimeError("Not implemented")
    # -- Prove
    correct: bool = False
    # -- Return
    return correct

# -- DSP for Lean

def single_proof_search_dsp_lean(
        eng: Engine,
        server: Server,
        data_pt: dict,
    ) -> bool:
    # -- Draft: [y_nl_pred_draft]_n ~ draft(eng, x_nl_prob, P_draft)
    y_nl_pred_drafts = draft(eng, data_pt)

    # -- Sketch: z_fl_pred_sketch ~ sketch(eng, x_nl_prob, [y_nl_pred_draft]_n, x_fl_prob, P_sketch)
    z_fl_pred_sketches, x_fl_prob = sketch(eng, data_pt, y_nl_pred_drafts)

    # -- Prove: y_fl = prove(eng, x_fl_prob, z_fl_pred_sketches)
    correct: bool = prove(eng, x_fl_prob, z_fl_pred_sketches)

    # -- Return
    return correct

def full_proof_search_dsp_lean(
    eng: Engine,
    server: Server,
    path_2_eval_dataset: Path,
):
    # -- Get eval data
    eval_dataset: list[dict] = json.load(open(path_2_eval_dataset, 'r'))
    print(f'{len(eval_dataset)=}')
    # -- Proof search by DSP over all eval data
    for data_pt in tqdm(eval_dataset, total=len(eval_dataset), desc='DSP proof loop per data point in benchmark.'):
        print(f'{data_pt=}')
        flag = single_proof_search_dsp_lean(eng, server, data_pt)
    return

experiment_dir = Path(__file__).resolve().parent

# -- Main

def main(
    path_2_eval_dataset: str = experiment_dir / 'debug/toy_example1_dsp/dsp_debug5_sf/dsp_debug5_sf_train.json',
    # model: str = 'deepseek-ai/deepseek-math-7b-instruct',
    # model: str = 'gpt2',
    # model: str = 'gpt-3.5-turbo',
    model: str = 'gpt-4o',
    start: int = 0,
    end: int = sys.maxsize,
    # end: int = 10,  # do 10 so enough boxed qs are there
    batch_size: int = 10,  # putnam has 348
    n_samples: int = 1, # num seqs to return for given prompt
    max_tokens: int = 2048,
    top_p: float = 0.95,
    temperature: float = 0.8,
    **kwargs,
):
    import time
    start_time = time.time()
    path_2_eval_dataset = Path(path_2_eval_dataset).expanduser()
    print(f'{path_2_eval_dataset=}')

    server = Server()

    # - Start wandb run
    # print(f'\n\n-- Setup params')
    # CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
    # current_tmux_session = os.environ.get("TMUX", "").split(",")[-1]
    # today = datetime.datetime.now().strftime("%Y-m%m-d%d-t%Hh_%Mm_%Ss")
    # config = {'today': today, "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES, "current_tmux_session": current_tmux_session, "model": model, "path_2_eval_dataset": path_2_eval_dataset}
    # project: str = 'pypantograph'
    # run_name = f"{project}: ({config})"
    # run = wandb.init(mode=mode, project=project, name=run_name, save_code=True, config=config)
    # print(f"{run.url=}")
    # print(f'\n Config: \n{config=}')

    # - Run DSP for Lean
    api_key = os.environ['OPENAI_API_KEY']
    draft_sampling_params = SamplingParams(n=n_samples, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=STOP_TOKENS_DRAFT_V0)
    sketch_sampling_params = SamplingParams(n=n_samples, max_tokens=max_tokens, top_p=top_p, temperature=temperature, stop=STOP_TOKENS_SKETCH_V0)
    eng: OpenAI_DSP_Engine = OpenAI_DSP_Engine(
        model=model,
        api_key=api_key,
        verbose_init=True,
        draft_sampling_params=draft_sampling_params,
        sketch_sampling_params=sketch_sampling_params,
    )

    # - Full proof search with DSP
    print(f'\n\n-- Full proof search with DSP')
    full_proof_search_dsp_lean(eng, server, path_2_eval_dataset)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")

    # - End run
    # wandb.config.update(config)
    # print(f"{wandb.config=}")
    # run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DSP',
        description="Draft-Sketch-Prove on Lean code"
    )
    parser.add_argument(
        'mode',
        help="Function",
        choices=['eval', 'prompts'],
    )
    parser.add_argument(
        "--eval-dataset",
        help="Evaluation dataset path",
        default=experiment_dir / 'debug/toy_example1_dsp/dsp_debug5_sf/dsp_debug5_sf_train.json',
    )
    parser.add_argument("--model", help="Model", default="gpt-4o")
    parser.add_argument("--start", default=0)
    parser.add_argument("--end", default=sys.maxsize)
    parser.add_argument("--batchsize", default=10)
    parser.add_argument("--n-samples", default=1)
    parser.add_argument("--max-tokens", default=2048)
    parser.add_argument("--top-p", default=0.95)
    parser.add_argument("--temperature", default=0.8)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.mode == "eval":
        main(**args)
    elif args.mode == "prompts":
        prompt = get_prompt_sketch_template_4_lean_v0(verbose=args.verbose)
        print(prompt)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
