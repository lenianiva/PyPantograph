import sys, os, json, subprocess, time, datetime
from abc import abstractmethod
from pathlib import Path
from dataclasses import asdict
from typing import Union, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI
import wandb
from tenacity import retry, stop_after_attempt, wait_exponential
from pantograph import Server, ServerError, DEFAULT_CORE_OPTIONS
from pantograph.search import SearchResult
from termcolor import colored

from solve.prompts import (
    extract_lean_code,
    SYSTEM_PROMPT_DRAFT_V0,
    SYSTEM_PROMPT_SKETCH_V0,
    prompt_draft_template_lean4_v0,
    prompt_sketch_template_lean4_v0,
    STOP_TOKENS_DRAFT_V0,
    STOP_TOKENS_SKETCH_V0,
    get_prompt_sketch_template_4_lean_v0,
)
from solve.prove import HammerAgent
from solve.data import (
    Datum,
    SamplingParams,
    SketchParseFailure,
    SearchFailure,
    DatumResult,
)

# prompt_draft_template_lean4_v0 = "Draft an informal solution similar to the one below. The informal solution will be used to sketch a formal proof in the Lean 4 Proof Assistant. Here are some examples of informal problem solutions pairs:\n\nInformal:\n(*### Problem\n\nProve that for any natural number n, n + 0 = n.\n\n### Solution\n\nConsider any natural number n. From properties of addition, adding zero does not change its values. Thus, n + 0 = n.*)\n\nInformal:\n(*### Problem\n\nProve that for any natural number n, n + (m + 1) = (n + m) + 1.\n\n### Solution\n\nConsider any natural numbers n and m. From properties of addition, adding 1 to the sum of n and m is the same as first adding m to n and then adding 1. Thus, n + (m + 1) = (n + m) + 1.*)\n\nInformal:\n(*### Problem\n\nProve that for any natural number n and m, n + m = m + n.\n\n### Solution\n\nConsider any natural numbers n and m. We will do induction on n. Base case: 0 + m = m + 0 by properties of addition. Inductive step, we have n + m = m + n. Then (n + 1) + m = (n + m) + 1 = (m + n) + 1 = m + (n + 1). Thus, by induction, n + m = m + n, qed.*)\n\nInformal: \n(*### Problem\n\n{nl_problem}\n\n### Solution\n"


class Engine:
    def __init__(self):
        pass

    def __call__(self, *args, **kwards):
        pass

    @abstractmethod
    def sample_draft(self, prompt: str):
        pass
    @abstractmethod
    def sample_sketch(self, prompt: str):
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
        print(f'{base_url=}') if verbose_init else None


        if not ('gpt-4-' in model or 'gpt-3.5-' in model or 'gpt-4o' in model or model == "o1-preview"):
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

    @property
    def role_prompt(self) -> str:
        return "assistant" if self.model.startswith("o1") else "system"

    def sample_draft(self, prompt: str):
        extra = {} if self.model.startswith("o1") else dict(
            seed=0,
            temperature=self.draft_sampling_params.temperature,
            top_p=self.draft_sampling_params.top_p,
            stop=self.draft_sampling_params.stop[:3],
        )
        return self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": self.role_prompt, "content": self.draft_system_prompt},
                {"role": "user", "content": prompt},
            ],
            n=self.draft_sampling_params.n,
            **extra,
        )
    def sample_sketch(self, prompt: str):
        extra = {} if self.model.startswith("o1") else dict(
            seed=0,
            temperature=self.sketch_sampling_params.temperature,
            top_p=self.sketch_sampling_params.top_p,
        )
        return self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": self.role_prompt, "content": self.sketch_system_prompt},
                {"role": "user", "content": prompt},
            ],
            n=self.sketch_sampling_params.n,
            **extra,
            # stop=eng.sketch_sampling_params.stop[:3],
        )

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def autoformalize_prob(
        eng: Engine,
        datum: Datum,
        verbose: bool = False,
    ):
    """ Autoformalize natural language problem to formal language problem. """
    pass

#@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def step_draft(
        eng: Engine,
        datum: Datum,
        verbose: bool = False,
    ) -> list:
    """
    Creates (informal nl) draft (nl soln, nl proof sketch) for latter use in a formal proof sketch.
        y_pred_nl ~ draft(eng, x_nl_prob, P_draft)
    """
    # Make prompt from template
    nl_problem: str = datum.nl_problem_str
    prompt = eng.draft_prompt_template.replace('{nl_problem}', nl_problem)
    # Get all **completions** to single prompt, one (in) -> many (out)
    # ref: https://platform.openai.com/docs/api-reference/chat/object
    response: Any = eng.sample_draft(prompt)
    # Get all completions for single prompt
    completions: list[str] = [
        completion.message.content
        for completion in response.choices
    ]  # response.choices[i].message
    drafts: list[str] = completions
    return drafts

#@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, max=128))
def step_sketch(
        eng: Engine,
        datum: Datum,
        drafts: list[str],
        autoformalize_prob_in_prompt: bool = False,
        verbose: bool = False,
    ) -> Tuple[list[str], str]:
    """
    Creates (formal fl) sketch (fl proof sketch) for latter use in a formal proof sketch.
        z_pred_fl ~ sketch(eng, x_nl_prob, y_pred_nl, x_fl_prob, P_sketch)
    """
    assert len(drafts) == 1, f"For now only 1 draft."
    # Make prompt from template
    x_nl_problem: str = datum.nl_problem_str
    y_nl_solution: str = drafts[0]
    x_fl_problem = None
    if autoformalize_prob_in_prompt:
        # prompt = eng.sketch_prompt_template.replace('{nl_problem}', x_nl_problem).replace('{nl_solution}', y_nl_solution)
        not NotImplemented
    else:
        x_fl_problem = datum.fl_problem if datum.fl_problem else autoformalize_prob(eng, datum)
        prompt = eng.sketch_prompt_template.replace('{fl_problem}', x_nl_problem).replace('{fl_problem}', y_nl_solution)
    # Get all **completions** to single prompt, one (in) -> many (out), ref: https://platform.openai.com/docs/api-reference/chat/object
    response: Any = eng.sample_sketch(prompt)
    # Get all completions for single prompt
    completions: list[str] = [completion.message.content for completion in response.choices]  # response.choices[i].message
    sketches: list[str] = completions
    # Return
    return sketches, x_fl_problem

def step_prove(
        eng: Engine,
        server: Server,
        fl_prob: str,
        fl_sketch: str,
    ) -> Union[SketchParseFailure, SearchFailure, SearchResult]:
    """

    Complete formal sketch and check if it proves the theorem.

    fl_prob --> Lean4 theorem (problem)
    fl_sketch --> Lean4 Form Sketch --> have x have ha

    """
    # If this throws index out of bound errors it means the source doesn't contain walled off Lean sections.
    print(colored("Sketch:", "yellow"), fl_sketch)
    lean_code = "\n".join(extract_lean_code(fl_sketch))
    print(colored("Lean code:", "light_grey", attrs=["bold"]), lean_code)

    try:
        states = server.load_sorry(lean_code)
    except ServerError as e:
        msg = f"Encountered exception: {e}"
        print(colored(msg, "red"))
        return SketchParseFailure(
            sketch=fl_sketch,
            error=msg,
        )

    if len(states) != 1:
        print(colored("Model must output one compilation unit", "red"))
        return SketchParseFailure(
            sketch=fl_sketch,
            error="Model must output one compilation unit",
        )

    state = states[0]

    if isinstance(state, list) and len(state) > 0:
        # This means `state` contains error messages
        msg = "\n".join(state)
        print(colored("Sketch failed:", "red"), msg)
        return SketchParseFailure(
            sketch=fl_sketch,
            error=f"Sketch failed: {msg}",
        )

    agent = HammerAgent()
    try:
        result = agent.search(
            server,
            state,
            max_steps=1000,
            max_trials_per_goal=len(agent.tactics) + 1,
        )
        print(colored(f"Result: {result}", "blue"))

        return result
    except Exception as e:
        return SearchFailure(
            error=f"Server threw exception",
            sketch=fl_sketch,
            message=str(e),
        )

# -- DSP for Lean

def single_proof_search_dsp_lean(
        eng: Engine,
        server_func,
        datum: Datum,
    ) -> DatumResult:

    start_time = time.time()
    # -- Draft: [y_nl_pred_draft]_n ~ draft(eng, x_nl_prob, P_draft)
    y_nl_pred_drafts = step_draft(eng, datum)

    # -- Sketch: z_fl_pred_sketch ~ sketch(eng, x_nl_prob, [y_nl_pred_draft]_n, x_fl_prob, P_sketch)
    z_fl_pred_sketches, x_fl_prob = step_sketch(eng, datum, y_nl_pred_drafts)

    assert len(z_fl_pred_sketches) == eng.sketch_sampling_params.n


    results = []
    success = False
    for i_sketch, sketch in enumerate(z_fl_pred_sketches):
        if len(z_fl_pred_sketches):
            print(colored(f"Sketch {1+i_sketch}/{len(z_fl_pred_sketches)}", attrs=["bold", "underline"]))

        try:
            server = server_func()
        except Exception as e:
            print(colored(f"Failed to create server: {e}", "red"))
            return DatumResult(
                name=str(datum),
                error=str(e),
            )
        # -- Prove: y_fl = prove(eng, x_fl_prob, z_fl_pred_sketches)
        prove_result = step_prove(eng, server, x_fl_prob, sketch)
        results.append(prove_result)
        if isinstance(prove_result, SearchResult) and prove_result.success:
            success = True
            break
    duration = time.time() - start_time

    return DatumResult(
        name=str(datum),
        success=success,
        proves=results,
        duration=duration,
    )

def full_proof_search_dsp_lean(
        eng: Engine,
        server_func,
        data: list[Datum],
        path_output: Path,
    ):
    n_success = 0
    n_tried = 0
    # -- Proof search by DSP over all eval data
    for i, datum in tqdm(enumerate(data), total=len(data), desc='DSP proof loop per data point in benchmark.'):
        output_path = path_output / f"{i:03}.json"
        key = str(datum)
        # Detect if file exists
        if output_path.is_file():
            obj = json.load(open(output_path, "r"))
            if obj['name'] != key:
                print(colored(f"Existing datum name {obj['name']} does not match dataset {key}. The output directory may be wrong"))
                return

            print(f"Skipped {output_path.name}:", colored(key, "green"))
            continue

        n_tried += 1
        print(f"Problem {output_path.name}:", colored(key, "cyan"))

        result = single_proof_search_dsp_lean(eng, server_func, datum)
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f)
        if result.success:
            n_success += 1
        #server.gc()
    print(f"Proved {n_success}/{n_tried} problems")


experiment_dir = Path(__file__).resolve().parent

def load_data(args) -> list[Datum]:
    p = Path(args.dataset).expanduser()
    data = None
    if p.suffix == ".json":
        data = [
            Datum.load(obj, data_format=args.format)
            for obj in json.load(open(p, 'r'))
        ]
    elif p.suffix == ".jsonl":
        with open(p, 'r') as f:
            data = [
                Datum.load(json.loads(line), data_format=args.format)
                for line in list(f)
            ]
    else:
        raise ValueError(f"Unknown data suffix: {p.suffix}")
    data = [datum for datum in data if datum]
    return data

# -- Main

def main(args):
    start_time = time.time()

    # Setup paths and data
    data_eval = load_data(args)
    path_output = Path(args.output)
    path_output.mkdir(exist_ok=True, parents=True)

    project_path = experiment_dir / 'lean_src_proj'

    print("Building src project ...")
    while True:
        # Lean sometimes fails to build. Try until it succeeds.
        completion = subprocess.run(['lake', 'build'], cwd=project_path, check=False)
        if completion.returncode == 0:
            break;
    print("Built src project")

    # Start server
    def server_func():
        return Server(
            imports=["Mathlib", "Aesop"],
            project_path=project_path,
            core_options=DEFAULT_CORE_OPTIONS,
        )

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
    draft_sampling_params = SamplingParams(
        n=1, #args.n_samples,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        stop=STOP_TOKENS_DRAFT_V0,
    )
    sketch_sampling_params = SamplingParams(
        n=args.n_samples,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        stop=STOP_TOKENS_SKETCH_V0,
    )
    eng: OpenAI_DSP_Engine = OpenAI_DSP_Engine(
        model=args.model,
        api_key=api_key,
        verbose_init=True,
        draft_sampling_params=draft_sampling_params,
        sketch_sampling_params=sketch_sampling_params,
    )

    print(colored(f"DSP on {len(data_eval)} points", "blue", attrs=["bold", "underline"]))
    print(f"Draft={draft_sampling_params}")
    print(f"Sketch={sketch_sampling_params}")
    # - Full proof search with DSP
    full_proof_search_dsp_lean(eng, server_func, data_eval, path_output)

    dt = datetime.timedelta(seconds=time.time() - start_time)
    print(colored(f"Time elapsed: {dt}", "magenta"))

    # - End run
    # wandb.config.update(config)
    # print(f"{wandb.config=}")
    # run.finish()

def check(args):
    path_output = Path(args.output)
    data = load_data(args)
    n_success = 0
    n_tried = 0
    for i, datum in tqdm(enumerate(data), total=len(data), desc='DSP proof loop per data point in benchmark.'):
        file_name = path_output / f"{i:03}.json"
        key = str(datum)
        # Detect if file exists
        obj = json.load(open(file_name, "r"))
        if obj['name'] != key:
            print(colored(f"Existing datum name {obj['name']} does not match dataset {key}. The output directory may be wrong", "red"))
            return

        n_tried += 1
        if obj['success']:
            n_success += 1
    print(f"Proved {n_success}/{n_tried} problems")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='DSP',
        description="Draft-Sketch-Prove on Lean code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'mode',
        help="Function",
        choices=['prompts', 'eval', 'check'],
    )
    parser.add_argument(
        "--dataset",
        help="Evaluation dataset path",
        default=experiment_dir / 'debug/toy_example1_dsp/dsp_debug5_sf/dsp_debug5_sf_train.json',
    )
    parser.add_argument(
        "--output",
        help="Result directory",
        default=experiment_dir / 'result',
    )
    parser.add_argument(
        "--model",
        help="Model",
        default="gpt-4o",
        choices=["gpt2", "gpt-3.5-turbo", "gpt-4o", "deepseek-ai/deepseek-math-7b-instruct", "o1-preview"],
    )
    parser.add_argument(
        "--format",
        help="Data format",
        default="default",
        choices=["default", "minif2f"],
    )
    parser.add_argument("--start", default=0)
    parser.add_argument("--end", default=sys.maxsize)
    parser.add_argument(
        "--batchsize",
        default=10, type=int,
        help="putnam has 348",
    )
    parser.add_argument(
        "--n-samples",
        default=1, type=int,
        help="Number of sketch samples for a draft",
    )
    parser.add_argument(
        "--max-tokens",
        default=2048, type=int,
        help="Maximum number of tokens in one sample",
    )
    parser.add_argument(
        "--top-p",
        default=0.95, type=float,
        help="Sampling top p via nucleus sampling",
    )
    parser.add_argument(
        "--temperature",
        default=0.8, type=float,
        help="Sampling temperature",
    )
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.mode == "prompts":
        prompt = get_prompt_sketch_template_4_lean_v0(verbose=args.verbose)
        print(prompt)
    elif args.mode == "eval":
        main(args)
    elif args.mode == 'check':
        check(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
