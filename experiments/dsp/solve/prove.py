import collections
from typing import Optional
from pantograph.search import Agent
from pantograph.expr import GoalState, Tactic

class HammerAgent(Agent):

    def __init__(self):
        super().__init__()

        self.goal_tactic_id_map = collections.defaultdict(lambda : 0)
        self.tactics = [
            "aesop",
        ]

    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
            informal_stmt: str,
            informal_proof: str) -> Optional[Tactic]:
        key = (state.state_id, goal_id)
        i = self.goal_tactic_id_map[key]

        if i >= len(self.tactics):
            return None

        self.goal_tactic_id_map[key] = i + 1
        return self.tactics[i]
