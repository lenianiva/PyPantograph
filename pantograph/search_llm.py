import search
from dataclasses import dataclass
from typing import override, Optional
import collections, unittest

from pantograph.server import Server, TacticFailure, ServerError
from pantograph.expr import Expr, Tactic, GoalState
from pantograph.gen_tactic import LEAN4_REWRITE, select_tactic

class LLMAgent(search.Agent):

    def __init__(self):
        super().__init__()
        self.n_trials = 5

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

    @override
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
        for i in range(self.n_trails):
            print(f"===============trail {str(i)}============")
            try:
                state = select_tactic.run(self.server, state, goal_id = 1)
                tactic, new_state = state.ret_value
                for m in state.messages():
                    print(m["role"], ":", m["content"])

                print("\n-- new state --\n", new_state)
                break
                
            except ServerError as e:
                print(f"server error: {e}")
                continue

        return tactics[i]
