# Goals and Tactics

Executing tactics in Pantograph is very simple.

```python
from pantograph import Server

server = Server()
state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
```

