"""
Class which manages a Pantograph instance. All calls to the kernel uses this
interface.
"""
import json, pexpect, pathlib, unittest
from pantograph.expr import Variable, Goal, GoalState, Tactic, TacticNormal

def _get_proc_cwd():
    return pathlib.Path(__file__).parent
def _get_proc_path():
    return _get_proc_cwd() / "pantograph"

class ServerError(Exception):
    pass

class Server:

    def __init__(self,
                 imports=["Init"],
                 options=[],
                 timeout=20,
                 maxread=1000000):
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.maxread = maxread
        self.proc_cwd = _get_proc_cwd()
        self.proc_path = _get_proc_path()

        self.options = options
        self.args = " ".join(imports + [f'--{opt}' for opt in options])
        self.proc = None
        self.restart()

    def restart(self):
        if self.proc is not None:
            self.proc.close()
        self.proc = pexpect.spawn(
            f"{self.proc_path} {self.args}",
            encoding="utf-8",
            maxread=self.maxread,
            cwd=self.proc_cwd,
        )
        self.proc.setecho(False)

    def run(self, cmd, payload):
        s = json.dumps(payload)
        self.proc.sendline(f"{cmd} {s}")
        try:
            self.proc.expect("{.*}\r\n", timeout=self.timeout)
            output = self.proc.match.group()
            return json.loads(output)
        except pexpect.exceptions.TIMEOUT:
            raise pexpect.exceptions.TIMEOUT

    def reset(self):
        return self.run("reset", {})

    def goal_start(self, expr: str) -> GoalState:
        result = self.run('goal.start', {"expr": str(expr)})
        if "error" in result:
            raise ServerError(result["desc"])
        return GoalState(state_id = result["stateId"], goals = [Goal.sentence(expr)])

    def goal_tactic(self, state: GoalState, goal_id: int, tactic: Tactic) -> GoalState:
        args = { "stateId": state.state_id, "goalId": goal_id}
        if isinstance(tactic, TacticNormal):
            args["tactic"] = tactic.payload
        else:
            raise Exception(f"Invalid tactic type: {tactic}")
        result = self.run('goal.tactic', args)
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        state_id = result["nextStateId"]
        goals = [Goal._parse(payload) for payload in result["goals"]]
        return GoalState(state_id, goals)

def get_version():
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"],
                          stdout=subprocess.PIPE,
                          cwd=_get_proc_cwd()) as p:
        return p.communicate()[0].decode('utf-8').strip()


class TestServer(unittest.TestCase):

    def test_version(self):
        self.assertEqual(get_version(), "0.2.14")

    def test_goal_start(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, goal_id=0, tactic=TacticNormal("intro a"))
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[Variable(name="a", t="Prop")],
            target="∀ (q : Prop), a ∨ q → q ∨ a",
            name=None,
        )])
        self.assertEqual(str(state1.goals[0]),"a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a")

if __name__ == '__main__':
    unittest.main()
