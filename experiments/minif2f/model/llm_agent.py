from typing import Optional
import collections, unittest
from termcolor import colored
from pantograph.search import Agent
from pantograph.server import Server, TacticFailure, ServerError
from pantograph.expr import Expr, Tactic, GoalState
from .gen_tactic import LEAN4_REWRITE, select_tactic
from .options import CORE_OPTIONS
import sglang as sgl

class LLMAgent(Agent):
    """
    A LLM-based proof agent from SGL
    """

    def __init__(self, server,
                 use_hammer=True,
                 use_llm=True,
                 feedback_turns=3):
        super().__init__()
        self.n_trials = 5
        self.server = server

        if use_llm:
            sgl.set_default_backend(sgl.OpenAI("gpt-4"))

        self.goal_tactic_id_map = collections.defaultdict(lambda : 0)

        self.use_hammer = use_hammer
        self.use_llm = use_llm
        self.feedback_turns = feedback_turns
        if use_hammer:
            self.tactics = [
                "aesop",
                #"simp",
                #"rfl",
                #"decide",
            ]
        else:
            self.tactics = []

        self.informal_stmt = ""
        self.informal_proof = ""

    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
        ) -> Optional[Tactic]:
        key = (state.state_id, goal_id)
        i = self.goal_tactic_id_map[key]

        target = state.goals[goal_id].target
        if i >= len(self.tactics) and not self.use_llm:
            return None
        elif i >= len(self.tactics):
            assert self.use_llm
            new_state = None
            for ii in range(self.n_trials):
                print(f"===============trail {str(ii)}============")
                s = select_tactic.run(
                    server=self.server,
                    state=state,
                    goal_id=goal_id,
                    informal_stmt=self.informal_stmt,
                    informal_proof=self.informal_proof,
                    feedback_turns=self.feedback_turns)
                tactic, new_state = s.ret_value
                for m in s.messages():
                    print(m["role"], ":", m["content"])

                print("\n-- new state --\n", new_state)
                if tactic:
                    if not isinstance(tactic, Tactic):
                        print(colored("[Tactic] Failed:", "red"), tactic)
                        return None
                    return tactic
            return None
        else:
            self.goal_tactic_id_map[key] = i + 1
            return self.tactics[i]

class TestSearch(unittest.TestCase):

    # def test_miniF2F(self):
    #     problem = {"id": "mathd_algebra_478",
    #                "split": "test",
    #                "formal_statement": "theorem mathd_algebra_478\n  (b h v : \u211d)\n  (h\u2080 : 0 < b \u2227 0 < h \u2227 0 < v)\n  (h\u2081 : v = 1 / 3 * (b * h))\n  (h\u2082 : b = 30)\n  (h\u2083 : h = 13 / 2) :\n  v = 65 := sorry",
    #                "header": "import Mathlib.Algebra.BigOperators.Basic\nimport Mathlib.Data.Real.Basic\nimport Mathlib.Data.Complex.Basic\nimport Mathlib.Data.Nat.Log\nimport Mathlib.Data.Complex.Exponential\nimport Mathlib.NumberTheory.Divisors\nimport Mathlib.Data.ZMod.Defs\nimport Mathlib.Data.ZMod.Basic\nimport Mathlib.Topology.Basic\nimport Mathlib.Data.Nat.Digits\n\nopen BigOperators\nopen Real\nopen Nat\nopen Topology",
    #                "informal_stmt": "The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.",
    #                "informal_proof": "We are given that $B = 30$ and $h = 6.5$ and asked to find $\\frac{1}{3}Bh$.  We find that \\[\\frac{1}{3}Bh = \\frac{1}{3}(30)(6.5) = (10)(6.5) = 65.\\]"}
    #     server = Server(imports=["Mathlib.Algebra.BigOperators.Basic", "Mathlib.Data.Real.Basic"])
    #     target = "∀ (b h v : ℝ)  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)  (h₁ : v = 1 / 3 * (b * h))  (h₂ : b = 30)  (h₃ : h = 13 / 2) , v = 65"
    #     # target = "theorem mathd_algebra_478\n  (b h v : ℝ)\n  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)\n  (h₁ : v = 1 / 3 * (b * h))\n  (h₂ : b = 30)\n  (h₃ : h = 13 / 2) :\n  v = 65 := sorry"
    #     agent = LLMAgent(server)
    #     flag = agent.search(server=server, target=target, verbose=True)
    #     self.assertTrue(flag)


    def test_solve(self):

        server = Server(core_options=CORE_OPTIONS)
        agent = LLMAgent(server, use_hammer=False)
        goal_state = server.goal_start("∀ (p q: Prop), p -> p")
        flag = agent.search(server=server, goal_state=goal_state, verbose=True)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server(core_options=CORE_OPTIONS)
        agent = LLMAgent(server, use_hammer=False)
        goal_state = server.goal_start("∀ (p q: Prop), Or p q -> Or q p")
        flag = agent.search(server=server, goal_state=goal_state, verbose=True)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
