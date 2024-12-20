"""
Class which manages a Pantograph instance. All calls to the kernel uses this
interface.
"""
import json, pexpect, unittest, os
from typing import Union
from pathlib import Path
from pantograph.expr import (
    parse_expr,
    Expr,
    Variable,
    Goal,
    GoalState,
    Tactic,
    TacticHave,
    TacticLet,
    TacticCalc,
    TacticExpr,
)
from pantograph.data import CompilationUnit

def _get_proc_cwd():
    return Path(__file__).parent
def _get_proc_path():
    return _get_proc_cwd() / "pantograph-repl"

def get_lean_path(project_path):
    """
    Extracts the `LEAN_PATH` variable from a project path.
    """
    import subprocess
    p = subprocess.check_output(
        ['lake', 'env', 'printenv', 'LEAN_PATH'],
        cwd=project_path,
    )
    return p

class TacticFailure(Exception):
    """
    Indicates a tactic failed to execute
    """
class ServerError(Exception):
    """
    Indicates a logical error in the server.
    """


DEFAULT_CORE_OPTIONS = ["maxHeartbeats=0", "maxRecDepth=100000"]


class Server:
    """
    Main interaction instance with Pantograph
    """

    def __init__(self,
                 imports=["Init"],
                 project_path=None,
                 lean_path=None,
                 # Options for executing the REPL.
                 # Set `{ "automaticMode" : False }` to handle resumption by yourself.
                 options={},
                 core_options=DEFAULT_CORE_OPTIONS,
                 timeout=120,
                 maxread=1000000):
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.project_path = project_path if project_path else _get_proc_cwd()
        if project_path and not lean_path:
            lean_path = get_lean_path(project_path)
        self.lean_path = lean_path
        self.maxread = maxread
        self.proc_path = _get_proc_path()

        self.options = options
        self.core_options = core_options
        self.args = " ".join(imports + [f'--{opt}' for opt in core_options])
        self.proc = None
        self.restart()

        # List of goal states that should be garbage collected
        self.to_remove_goal_states = []

    def is_automatic(self):
        """
        Check if the server is running in automatic mode
        """
        return self.options.get("automaticMode", True)

    def restart(self):
        if self.proc is not None:
            self.proc.close()
        env = os.environ
        if self.lean_path:
            env = env | {'LEAN_PATH': self.lean_path}

        self.proc = pexpect.spawn(
            f"{self.proc_path} {self.args}",
            encoding="utf-8",
            maxread=self.maxread,
            timeout=self.timeout,
            cwd=self.project_path,
            env=env,
        )
        self.proc.setecho(False) # Do not send any command before this.
        try:
            ready = self.proc.readline() # Reads the "ready."
            assert ready.rstrip() == "ready.", f"Server failed to emit ready signal: {ready}; Maybe the project needs to be rebuilt"
        except pexpect.exceptions.TIMEOUT as exc:
            raise RuntimeError("Server failed to emit ready signal in time") from exc

        if self.options:
            self.run("options.set", self.options)

        self.run('options.set', {'printDependentMVars': True})

    def run(self, cmd, payload):
        """
        Runs a raw JSON command. Preferably use one of the commands below.

        :meta private:
        """
        assert self.proc
        s = json.dumps(payload)
        self.proc.sendline(f"{cmd} {s}")
        try:
            line = self.proc.readline()
            try:
                obj = json.loads(line)
                if obj.get("error") == "io":
                    # The server is dead
                    self.proc = None
                return obj
            except Exception as e:
                self.proc.sendeof()
                remainder = self.proc.read()
                self.proc = None
                raise RuntimeError(f"Cannot decode: {line}\n{remainder}") from e
        except pexpect.exceptions.TIMEOUT as exc:
            self.proc = None
            return {"error": "timeout", "message": str(exc)}

    def gc(self):
        """
        Garbage collect deleted goal states to free up memory.
        """
        if not self.to_remove_goal_states:
            return
        result = self.run('goal.delete', {'stateIds': self.to_remove_goal_states})
        self.to_remove_goal_states.clear()
        if "error" in result:
            raise ServerError(result["desc"])

    def expr_type(self, expr: Expr) -> Expr:
        """
        Evaluate the type of a given expression. This gives an error if the
        input `expr` is ill-formed.
        """
        result = self.run('expr.echo', {"expr": expr})
        if "error" in result:
            raise ServerError(result["desc"])
        return parse_expr(result["type"])

    def goal_start(self, expr: Expr) -> GoalState:
        """
        Create a goal state with one root goal, whose target is `expr`
        """
        result = self.run('goal.start', {"expr": str(expr)})
        if "error" in result:
            print(f"Cannot start goal: {expr}")
            raise ServerError(result["desc"])
        return GoalState(
            state_id=result["stateId"],
            goals=[Goal.sentence(expr)],
            _sentinel=self.to_remove_goal_states,
        )

    def goal_tactic(self, state: GoalState, goal_id: int, tactic: Tactic) -> GoalState:
        """
        Execute a tactic on `goal_id` of `state`
        """
        args = {"stateId": state.state_id, "goalId": goal_id}
        if isinstance(tactic, str):
            args["tactic"] = tactic
        elif isinstance(tactic, TacticHave):
            args["have"] = tactic.branch
            if tactic.binder_name:
                args["binderName"] = tactic.binder_name
        elif isinstance(tactic, TacticLet):
            args["let"] = tactic.branch
            if tactic.binder_name:
                args["binderName"] = tactic.binder_name
        elif isinstance(tactic, TacticExpr):
            args["expr"] = tactic.expr
        elif isinstance(tactic, TacticCalc):
            args["calc"] = tactic.step
        else:
            raise RuntimeError(f"Invalid tactic type: {tactic}")
        result = self.run('goal.tactic', args)
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise TacticFailure(result["tacticErrors"])
        if "parseError" in result:
            raise TacticFailure(result["parseError"])
        return GoalState.parse(result, self.to_remove_goal_states)

    def goal_conv_begin(self, state: GoalState, goal_id: int) -> GoalState:
        """
        Start conversion tactic mode on one goal
        """
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": goal_id, "conv": True})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result, self.to_remove_goal_states)

    def goal_conv_end(self, state: GoalState) -> GoalState:
        """
        Exit conversion tactic mode on all goals
        """
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": 0, "conv": False})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result, self.to_remove_goal_states)

    def tactic_invocations(self, file_name: Union[str, Path]) -> list[CompilationUnit]:
        """
        Collect tactic invocation points in file, and return them.
        """
        result = self.run('frontend.process', {
            'fileName': str(file_name),
            'invocations': True,
            "sorrys": False,
            "newConstants": False,
        })
        if "error" in result:
            raise ServerError(result["desc"])

        units = [CompilationUnit.parse(payload) for payload in result['units']]
        return units

    def load_sorry(self, content: str) -> list[CompilationUnit]:
        """
        Executes the compiler on a Lean file. For each compilation unit, either
        return the gathered `sorry` s, or a list of messages indicating error.
        """
        result = self.run('frontend.process', {
            'file': content,
            'invocations': False,
            "sorrys": True,
            "newConstants": False,
        })
        if "error" in result:
            raise ServerError(result["desc"])

        units = [
            CompilationUnit.parse(payload, goal_state_sentinel=self.to_remove_goal_states)
            for payload in result['units']
        ]
        return units

    def env_add(self, name: str, t: Expr, v: Expr, is_theorem: bool = True):
        result = self.run('env.add', {
            "name": name,
            "type": t,
            "value": v,
            "isTheorem": is_theorem,
        })
        if "error" in result:
            raise ServerError(result["desc"])
    def env_inspect(
            self,
            name: str,
            print_value: bool = False,
            print_dependency: bool = False) -> dict:
        result = self.run('env.inspect', {
            "name": name,
            "value": print_value,
            "dependency": print_dependency,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        return result

    def env_save(self, path: str):
        result = self.run('env.save', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])
    def env_load(self, path: str):
        result = self.run('env.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])

    def goal_save(self, goal_state: GoalState, path: str):
        result = self.run('goal.save', {
            "id": goal_state.state_id,
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])
    def goal_load(self, path: str) -> GoalState:
        result = self.run('goal.load', {
            "path": path,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        state_id = result['id']
        result = self.run('goal.print', {
            'stateId': state_id,
            'goals': True,
        })
        if "error" in result:
            raise ServerError(result["desc"])
        return GoalState.parse_inner(state_id, result['goals'], self.to_remove_goal_states)


def get_version():
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"],
                          stdout=subprocess.PIPE,
                          cwd=_get_proc_cwd()) as p:
        return p.communicate()[0].decode('utf-8').strip()
