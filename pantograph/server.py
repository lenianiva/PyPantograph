"""
Class which manages a Pantograph instance. All calls to the kernel uses this
interface.
"""
import json, pexpect, pathlib, unittest
from pantograph.expr import Variable, Goal, GoalState, \
    Tactic, TacticHave, TacticCalc

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
        args = {"stateId": state.state_id, "goalId": goal_id}
        if isinstance(tactic, str):
            args["tactic"] = tactic
        elif isinstance(tactic, TacticHave):
            args["have"] = tactic.branch
        elif isinstance(tactic, TacticCalc):
            args["calc"] = tactic.step
        else:
            raise RuntimeError(f"Invalid tactic type: {tactic}")
        result = self.run('goal.tactic', args)
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result)

    def goal_conv_begin(self, state: GoalState, goal_id: int) -> GoalState:
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": goal_id, "conv": True})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result)

    def goal_conv_end(self, state: GoalState) -> GoalState:
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": 0, "conv": False})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result)


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
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[Variable(name="a", t="Prop")],
            target="∀ (q : Prop), a ∨ q → q ∨ a",
            name=None,
        )])
        self.assertEqual(str(state1.goals[0]),"a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a")

    def test_conv_calc(self):
        server = Server()
        state0 = server.goal_start("∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b")

        variables = [
            Variable(name="a", t="Nat"),
            Variable(name="b", t="Nat"),
            Variable(name="h", t="b = 2"),
        ]
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
        state2 = server.goal_tactic(state1, goal_id=0, tactic=TacticCalc("1 + a + 1 = a + 1 + 1"))
        self.assertEqual(state2.goals, [
            Goal(
                variables,
                target="1 + a + 1 = a + 1 + 1",
                name='calc',
            ),
            Goal(
                variables,
                target="a + 1 + 1 = a + b",
            ),
        ])
        state_c1 = server.goal_conv_begin(state2, goal_id=0)
        state_c2 = server.goal_tactic(state_c1, goal_id=0, tactic="rhs")
        state_c3 = server.goal_tactic(state_c2, goal_id=0, tactic="rw [Nat.add_comm]")
        state_c4 = server.goal_conv_end(state_c3)
        state_c5 = server.goal_tactic(state_c4, goal_id=0, tactic="rfl")
        self.assertTrue(state_c5.is_solved)

        state3 = server.goal_tactic(state2, goal_id=1, tactic=TacticCalc("_ = a + 2"))
        state4 = server.goal_tactic(state3, goal_id=0, tactic="rw [Nat.add_assoc]")
        self.assertTrue(state4.is_solved)


if __name__ == '__main__':
    unittest.main()