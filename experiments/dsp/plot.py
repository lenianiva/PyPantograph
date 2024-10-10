import json
from pathlib import Path
from matplotlib import pyplot
from typing import Callable, Optional
from tqdm import tqdm
import seaborn
import pandas
from termcolor import colored
from solve.data import (
    Datum,
    SamplingParams,
    SketchParseFailure,
    SearchFailure,
    DatumResult,
)

experiment_dir = Path(__file__).resolve().parent

def read_data(name: str, result_path: Path):
    assert result_path.is_dir(), f"Result path is not a directory: {result_path}"
    objs = [
        DatumResult.parse(json.load(open(file_name, "r")))
        for file_name in tqdm(result_path.glob("*"), desc="Reading results")
    ]

    print(colored(f"{name}", attrs=["underline"]))
    # Calculate the metrics
    successes = sum(obj.success for obj in objs)
    success_rate = successes / len(objs)
    print(f"Success rate: {successes} / {len(objs)} = {success_rate:.3f}")

    # Calculate hammer rates
    hammer_invocations = [
        obj.hammer_invocations
        for obj in objs
        if obj.hammer_invocations
    ]
    if hammer_invocations:
        avg_hammer_invocation = sum(hammer_invocations) / len(hammer_invocations)
        print(f"Hammer invocations: {avg_hammer_invocation:.3f}")
    else:
        print("Hammer invocations cannot be calculated")

    durations = [obj.duration for obj in objs if obj.duration and obj.duration > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"Durations: {avg_duration:.3f}")
    else:
        print("Durations cannot be calculated")
    return objs

def differentiated_histogram(
        path: Path,
        data: dict[str, list[dict]],
        key: str,
        key_func: Callable[[DatumResult], Optional[float]],
        xticks: list[float] = None,
        **kwargs,
    ):
    """
    Map objects using `key_func`, filtering o
    """
    dfs = [
        pandas.DataFrame({ 'name':  name, key: [ key_func(obj) for obj in objs ]})
        for name, objs in data.items()
    ]
    df = pandas.concat(dfs).reset_index()
    fig, ax = pyplot.subplots(figsize=(6,4))
    seaborn.histplot(
        df,
        ax=ax,
        x=key, hue="name",
        multiple="stack",
        **kwargs,
    )
    if xticks:
        ax.set_xticks(xticks, labels=[str(t) for t in xticks])
    fig.savefig(path, dpi=300)


def plot(args):

    assert len(args.result) == len(args.names), "Names must have a 1-1 correspondence with results"

    print(colored("Reading data ...", color="blue"))
    data = {
        name: read_data(name, Path(result_path))
        for name, result_path in zip(args.names, args.result)
    }

    path_plot_output = Path(args.plot_output)
    path_plot_output.mkdir(exist_ok=True, parents=True)
    # Generate plots
    differentiated_histogram(
        path_plot_output / "hammer.jpg",
        data,
        key="Hammered Goals",
        key_func=lambda obj: obj.hammer_invocations)
    differentiated_histogram(
        path_plot_output / "runtime.jpg",
        data,
        key="Runtime (s)",
        key_func=lambda obj: obj.duration if obj.duration > 0.0 else None,
        log_scale=True,
        xticks=[5, 10, 20, 40, 80, 160, 320],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='DSP Plot',
        description="Generates plots for DSP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result",
        help="Result directory",
        nargs="+",
        default=experiment_dir / 'result',
    )
    parser.add_argument(
        "--names",
        help="Experiment names",
        nargs="+",
        default=["unnamed"],
    )
    parser.add_argument(
        "--plot-output",
        help="Plot generation directory",
        default=experiment_dir / 'result-plot',
    )
    parser.add_argument(
        "--palette",
        help="Colour palette",
        default="Paired",
    )
    args = parser.parse_args()

    seaborn.set_palette(args.palette)

    plot(args)
