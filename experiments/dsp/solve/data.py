from typing import Union, Optional
from dataclasses import dataclass, field
from pantograph.search import SearchResult

@dataclass
class Datum:
    """
    Represents one theorem proving datapoint.
    """

    id: Optional[str] = None

    # Problem and solution in natural language
    nl_problem: Optional[Union[str, list[str]]] = None
    nl_solution: Optional[Union[str, list[str]]] = None

    # Problem in formal language
    fl_problem: Optional[str] = None

    def __str__(self):
        if self.id:
            return self.id
        return self.nl_problem_str

    @property
    def nl_problem_str(self) -> Optional[str]:
        if not self.nl_problem:
            return None
        if isinstance(self.nl_problem, list):
            return "\n".join(self.nl_problem)
        return self.nl_problem

    @staticmethod
    def load_default(obj: dict):
        """
        Loads data in the "default" format
        """
        fl_problem = obj.get("fl_problem")
        if isinstance(fl_problem, list):
            fl_problem = "\n".join(fl_problem)
        return Datum(
            nl_problem=obj.get("nl_problem"),
            nl_solution=obj.get("nl_solution"),
            fl_problem=fl_problem,
        )

    @staticmethod
    def load_minif2f(obj: dict):
        """
        Loads minif2f data
        """
        fl_problem = obj["formal_statement"].strip()
        if fl_problem.startswith("--"):
            return None
        return Datum(
            id=obj["id"],
            fl_problem=fl_problem,
            #header=obj["header"],
            nl_problem=obj["informal_stmt"],
            nl_solution=obj["informal_proof"],
        )

    @staticmethod
    def load(obj: dict, data_format: str):
        if data_format == "default":
            return Datum.load_default(obj)
        elif data_format == "minif2f":
            return Datum.load_minif2f(obj)
        else:
            raise ValueError(f"Invalid data format {data_format}")


@dataclass
class SamplingParams:
    n: int
    max_tokens: int
    top_p: int
    temperature: float
    stop: str

@dataclass(frozen=True)
class SketchParseFailure:
    error: str
    sketch: str
@dataclass(frozen=True)
class SearchFailure:
    error: str
    sketch: str
    message: str

@dataclass(frozen=True)
class DatumResult:
    """
    Result from one DSP data point
    """
    name: str
    error: Optional[str] = None
    duration: float = -1.0
    success: Optional[bool] = False
    proves: list[Union[SearchResult, SearchFailure, SketchParseFailure]] = field(default_factory=list)

    @staticmethod
    def parse_result(obj: dict):
        if "message" in obj:
            return SearchFailure(**obj)

        if "error" in obj:
            return SketchParseFailure(**obj)

        return SearchResult(**obj)

    @staticmethod
    def parse(obj: dict):
        return DatumResult(
            name=obj['name'],
            error=obj.get('error'),
            duration=obj.get('duration'),
            success=obj['success'],
            proves=[DatumResult.parse_result(o) for o in obj['proves']]
        )

    @property
    def hammer_invocations(self) -> Optional[float]:
        """
        Average number of hammer invocations required
        """
        li = [
            sr.n_goals_root
            for sr in self.proves
            if isinstance(sr, SearchResult)
        ]
        if not li:
            return None
        return sum(li)
