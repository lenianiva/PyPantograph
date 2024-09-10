"""
Class which manages a Pantograph instance. All calls to the kernel uses this
interface.
"""
import json, pexpect, pathlib, unittest, os
from pantograph.expr import parse_expr, Expr, Variable, Goal, GoalState, \
    Tactic, TacticHave, TacticCalc
from pantograph.compiler import TacticInvocation

def _get_proc_cwd():
    return pathlib.Path(__file__).parent
def _get_proc_path():
    return _get_proc_cwd() / "pantograph-repl"

class ServerError(Exception):
    pass

class Server:

    def __init__(self,
                 imports=["Init"],
                 project_path=None,
                 lean_path=None,
                 # Options for executing the REPL.
                 # Set `{ "automaticMode" : False }` to handle resumption by yourself.
                 options={},
                 core_options=[],
                 timeout=20,
                 maxread=1000000):
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.project_path = project_path if project_path else _get_proc_cwd()
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
        ready = self.proc.readline() # Reads the "ready."
        assert ready == "ready.\r\n"

        if self.options:
            self.run("options.set", self.options)

    def run(self, cmd, payload):
        """
        Runs a raw JSON command. Preferably use one of the commands below.
        """
        s = json.dumps(payload)
        self.proc.sendline(f"{cmd} {s}")
        try:
            line = self.proc.readline()
            return json.loads(line)
        except pexpect.exceptions.TIMEOUT as exc:
            raise exc

    def gc(self):
        """
        Garbage collect deleted goal states.

        Must be called periodically.
        """
        if self.to_remove_goal_states:
            self.run('goal.delete', {'stateIds': self.to_remove_goal_states})
            self.to_remove_goal_states.clear()

    def expr_type(self, expr: str) -> Expr:
        """
        Evaluate the type of a given expression. This gives an error if the
        input `expr` is ill-formed.
        """
        result = self.run('expr.echo', {"expr": expr})
        if "error" in result:
            raise ServerError(result["desc"])
        return parse_expr(result["type"])

    def goal_start(self, expr: str) -> GoalState:
        result = self.run('goal.start', {"expr": str(expr)})
        if "error" in result:
            raise ServerError(result["desc"])
        return GoalState(state_id=result["stateId"], goals=[Goal.sentence(expr)], _sentinel=self.to_remove_goal_states)

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
        return GoalState.parse(result, self.to_remove_goal_states)

    def goal_conv_begin(self, state: GoalState, goal_id: int) -> GoalState:
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": goal_id, "conv": True})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result, self.to_remove_goal_states)

    def goal_conv_end(self, state: GoalState) -> GoalState:
        result = self.run('goal.tactic', {"stateId": state.state_id, "goalId": 0, "conv": False})
        if "error" in result:
            raise ServerError(result["desc"])
        if "tacticErrors" in result:
            raise ServerError(result["tacticErrors"])
        if "parseError" in result:
            raise ServerError(result["parseError"])
        return GoalState.parse(result, self.to_remove_goal_states)

    def compile_unit(self, module: str) -> tuple[list[str], list[TacticInvocation]]:
        file_path = self.project_path / (module.replace('.', '/') + '.lean')
        result = self.run('compile.unit', {
            'module': module,
            'compilationUnits': True,
            'invocations': True
        })
        if "error" in result:
            raise ServerError(result["desc"])

        with open(file_path, 'rb') as f:
            content = f.read()
            units = [content[begin:end].decode('utf-8') for begin,end in result['units']]

        invocations = [TacticInvocation.parse(i) for i in result['invocations']]
        return units, invocations



def get_version():
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"],
                          stdout=subprocess.PIPE,
                          cwd=_get_proc_cwd()) as p:
        return p.communicate()[0].decode('utf-8').strip()


class TestServer(unittest.TestCase):

    def test_version(self):
        self.assertEqual(get_version(), "0.2.19")

    def test_expr_type(self):
        server = Server()
        t = server.expr_type("forall (n m: Nat), n + m = m + n")
        self.assertEqual(t, "Prop")

    def test_goal_start(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[Variable(name="a", t="Prop")],
            target="∀ (q : Prop), a ∨ q → q ∨ a",
            name=None,
        )])
        self.assertEqual(str(state1.goals[0]),"a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a")

        del state0
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

        state0b = server.goal_start("forall (p: Prop), p -> p")
        del state0b
        self.assertEqual(len(server.to_remove_goal_states), 1)
        server.gc()
        self.assertEqual(len(server.to_remove_goal_states), 0)

    def test_automatic_mode(self):
        server = Server()
        state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
        self.assertEqual(len(server.to_remove_goal_states), 0)
        self.assertEqual(state0.state_id, 0)
        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
        self.assertEqual(state1.state_id, 1)
        self.assertEqual(state1.goals, [Goal(
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h", t="a ∨ b"),
            ],
            target="b ∨ a",
            name=None,
        )])
        state2 = server.goal_tactic(state1, goal_id=0, tactic="cases h")
        self.assertEqual(state2.goals, [
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            ),
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="b"),
                ],
                target="b ∨ a",
                name="inr",
            ),
        ])
        state3 = server.goal_tactic(state2, goal_id=1, tactic="apply Or.inl")
        state4 = server.goal_tactic(state3, goal_id=0, tactic="assumption")
        self.assertEqual(state4.goals, [
            Goal(
                variables=[
                    Variable(name="a", t="Prop"),
                    Variable(name="b", t="Prop"),
                    Variable(name="h✝", t="a"),
                ],
                target="b ∨ a",
                name="inl",
            )
        ])


    def test_conv_calc(self):
        server = Server(options={"automaticMode": False})
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
