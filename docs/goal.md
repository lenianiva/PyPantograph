# Goals and Tactics

Executing tactics in Pantograph is very simple. To start a proof, call the
`Server.goal_start` function and supply an expression.

```python
from pantograph import Server

server = Server()
state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
```

This creates a *goal state*, which consists of a finite number of goals. In this
case since it is the beginning of a state, it has only one goal. `print(state0)` gives

```
GoalState(state_id=0, goals=[Goal(variables=[], target='forall (p : Prop), p -> p', name=None, is_conversion=False)], _sentinel=[])
```

To execute a tactic on a goal state, use `Server.goal_tactic`. This function
takes a goal id and a tactic. Most Lean tactics are strings.

```python
state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
```

