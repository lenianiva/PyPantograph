from abc import abstractmethod
import time
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
        self.trials = [0 for _ in self.state.goals]

    @property
    def next_goal_id(self) -> int:
        goal_id, _ = max(
            ((i, prio) for i, prio in enumerate(self.priorities) if not self.solved[i]),
            key=lambda x: x[1])
        return goal_id

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_solved(self) -> bool:
        return all(self.solved)

@dataclass(frozen=True)
class SearchResult:

    n_goals_root: int
    duration: float
    success: bool
    steps: int

class Agent:
    """
    An agent interface for proof search
    """
    tactic_feedback: Optional[str] = None

    @abstractmethod
    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
        ) -> Optional[Tactic]:
        """
        Implement this function to generate the next tactic for a goal
        """

    @abstractmethod
    def guidance(self, state: GoalState) -> list[float]:
        """
        Return a list of priorities determining which goal should be searched
        first. This will not be called on states with one or zero goals.
        """
        return [0.0 for _ in state.goals]
    @abstractmethod
    def reset(self):
        """
        Called after search
        """

    def search(self,
               server: Server,
               goal_state: GoalState,
               max_steps: int = 100,
               max_trials_per_goal: int = 5,
               verbose: bool = False) -> SearchResult:
        """
        Executes proof search on this state
        """

        assert server.is_automatic(), "Search must be run in automatic mode"

        n_goals_root = len(goal_state.goals)
        time_start = time.time()

        initial_state = SearchState(
            state=goal_state,
            parent=None,
            parent_goal_id=None,
            priorities=[0.0 for _ in goal_state.goals]
        )
        search_stack = [initial_state]
        for i_step in range(max_steps):
            assert search_stack, "No states in search stack"

            if verbose:
                print(f"I={i_step}: len(S) = {len(search_stack)}")
            search_state = search_stack[-1]

            assert isinstance(search_state, SearchState)

            if search_state.is_solved:
                return SearchResult(
                    n_goals_root=n_goals_root,
                    duration=time.time() - time_start,
                    success=True,
                    steps=i_step,
                )

            # Find the unsolved goal with the highest priority
            goal_id = search_state.next_goal_id

            if search_state.trials[goal_id] > max_trials_per_goal:
                # force halt the search
                tactic = None
            else:
                # Generate tactic for this goal
                tactic = self.next_tactic(search_state.state, goal_id)

            if verbose:
                print(f"Next tactic: {tactic}")
            if not tactic:
                # resets the feedback
                self.tactic_feedback = None
                # pop the current state and continue to the next
                search_stack.pop(-1)
                if not search_stack:
                    if verbose:
                        print("Search stack has been exhausted")
                    self.reset()
                    return SearchResult(
                        n_goals_root=n_goals_root,
                        duration=time.time() - time_start,
                        success=False,
                        steps=i_step,
                    )
                continue

            try:
                search_state.trials[goal_id] += 1
                state = search_state.state
                if verbose:
                    print(f"{state.state_id}.{goal_id}: {tactic} on {search_state.state.goals[goal_id]}")
                next_goal_state = server.goal_tactic(search_state.state, goal_id, tactic)
                # Generate priorities for the next goal state
                priorities = [0.0 for _ in next_goal_state.goals] \
                    if len(next_goal_state.goals) <= 1 else \
                    self.guidance(next_goal_state)
                parent = len(search_stack) - 1
                next_state = SearchState(
                    state=next_goal_state,
                    parent=parent,
                    parent_goal_id=goal_id,
                    priorities=priorities
                )
                search_stack.append(next_state)

            except TacticFailure as t:
                if verbose:
                    print(f"Tactic failed: {t}")
                self.tactic_feedback = str(t)
                # try the next tactic. this one failed

        if verbose:
            print("Search iteration limit exhausted")

        self.reset()
        return SearchResult(
            n_goals_root=n_goals_root,
            duration=time.time() - time_start,
            success=False,
            steps=max_steps,
        )


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

    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
    ) -> Optional[Tactic]:
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
        goal_state = server.goal_start("∀ (p q: Prop), p -> p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            verbose=False)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server()
        agent = DumbAgent()
        goal_state = server.goal_start("∀ (p q: Prop), Or p q -> Or q p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            verbose=False)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
