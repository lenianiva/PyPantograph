from dataclasses import dataclass
from typing import  Optional
import collections, unittest

from pantograph.server import Server, TacticFailure
from pantograph.expr import Expr, Tactic, GoalState


@dataclass
class SearchState:

    state: GoalState
    parent: Optional[int]
    parent_goal_id: Optional[int]
    priorities: list[float]

    def __post_init__(self):
        assert len(self.priorities) == len(self.state.goals)
        self.solved = [False for _ in self.state.goals]

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_solved(self) -> bool:
        return all(self.solved)


class Agent:

    def next_tactic(self, state: GoalState, goal_id: int) -> Optional[Tactic]:
        """
        Implement this function to generate the next tactic for a goal
        """

    def guidance(self, state: GoalState) -> list[float]:
        """
        Return a list of priorities determining which goal should be searched
        first. This will not be called on states with one or zero goals.
        """
        return [0.0 for _ in state.goals]
    def reset(self):
        """
        Called after search
        """

    def search(self,
               server: Server,
               target: Expr,
               max_steps: int = 1000,
               verbose: bool = False) -> bool:

        search_stack = [SearchState(state=server.goal_start(target),
                                    parent=None,
                                    parent_goal_id=None,
                                    priorities=[0.0])]
        """
        Executes proof search on this state
        """
        for i_step in range(max_steps):

            assert search_stack, "No states in search stack"

            if verbose:
                print(f"I={i_step}: len(S) = {len(search_stack)}")
            search_state = search_stack[-1]

            assert isinstance(search_state, SearchState)

            if search_state.is_solved:
                # if the state is solved, propagate this solved status
                if search_state.is_root:
                    self.reset()
                    return True

                search_stack.pop(-1)
                assert not search_stack[search_state.parent].solved[search_state.parent_goal_id]
                search_stack[search_state.parent].solved[search_state.parent_goal_id] = True
                continue

            # Find the unsolved goal with the highest priority
            goal_id, _ = max([(i, prio) for i, prio in enumerate(search_state.priorities) if not search_state.solved[i]],
                             key=lambda x:x[1])

            # Generate tactic for this goal
            tactic = self.next_tactic(search_state.state, goal_id)
            if not tactic:
                # pop the current state and continue to the next
                search_stack.pop(-1)
                if not search_stack:
                    if verbose:
                        print("Tactic list has been exhausted")
                    self.reset()
                    return False
                continue

            try:
                state = search_state.state
                if verbose:
                    print(f"{state.state_id}.{goal_id}: {tactic} on {search_state.state.goals[goal_id]}")
                next_goal_state = server.goal_tactic(search_state.state, goal_id, tactic)
                # Generate priorities for the next goal state
                priorities = [0.0 for _ in next_goal_state.goals] \
                    if len(next_goal_state.goals) <= 1 else \
                    self.guidance(next_goal_state)
                parent = len(search_stack) - 1
                search_stack.append(SearchState(state=next_goal_state,
                                                parent=parent,
                                                parent_goal_id=goal_id,
                                                priorities=priorities))

            except TacticFailure as t:
                print(f"Tactic failed: {t}")
                # try the next tactic. this one failed

        if verbose:
            print("Search iteration limit exhausted")

        self.reset()
        return False


class DumbAgent(Agent):

    def __init__(self):
        super().__init__()

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
        return tactics[i]


class TestSearch(unittest.TestCase):

    def test_solve(self):

        server = Server()
        agent = DumbAgent()
        flag = agent.search(server=server, target="∀ (p q: Prop), p -> p", verbose=True)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server()
        agent = DumbAgent()
        flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
