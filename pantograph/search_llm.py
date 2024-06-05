from typing import Optional
import collections, unittest
from pantograph.search import Agent
from pantograph.server import Server, TacticFailure, ServerError
from pantograph.expr import Expr, Tactic, GoalState
from pantograph.gen_tactic import LEAN4_REWRITE, select_tactic
import sglang as sgl 

class LLMAgent(Agent):

    def __init__(self, server):
        super().__init__()
        self.n_trials = 5
        self.server = server
        sgl.set_default_backend(sgl.OpenAI("gpt-4"))

        self.goal_tactic_id_map = collections.defaultdict(lambda : 0)
        self.intros = [
            "intro",
        ]
        self.tactics = [
            "intro h",
            "cases h",
            "apply Or.inl",
            "apply Or.inr",
        ]
        self.no_space_tactics = [
            "assumption",
        ]

    def next_tactic(self, state: GoalState, goal_id: int) -> Optional[Tactic]:
        key = (state.state_id, goal_id)
        i = self.goal_tactic_id_map[key]

        target = state.goals[goal_id].target
        if target.startswith('∀'):
            tactics = self.intros
        elif ' ' in target:
            tactics = self.tactics
        else:
            tactics = self.no_space_tactics

        if i >= len(tactics):
            return None

        self.goal_tactic_id_map[key] = i + 1
        new_state = None
        for ii in range(self.n_trials):
            print(f"===============trail {str(ii)}============")
            try:
                state = select_tactic.run(server = self.server, state=state, goal_id = goal_id)
                tactic, new_state = state.ret_value
                for m in state.messages():
                    print(m["role"], ":", m["content"])

                print("\n-- new state --\n", new_state)
                if tactic:
                    return tactic
                
            except ServerError as e:
                print(f"server error: {e}")
                continue
            except TacticFailure as e:
                print(f"tactic failure: {e}")
                continue
        

        return tactics[i]

class TestSearch(unittest.TestCase):

    def test_solve(self):

        server = Server()
        agent = LLMAgent(server)
        flag = agent.search(server=server, target="∀ (p q: Prop), p -> p", verbose=True)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server()
        agent = LLMAgent(server)
        flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
