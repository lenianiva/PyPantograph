# Proof Search

Inherit from the `pantograph.search.Agent` class to create your own search agent.
```python
from pantograph.search import Agent

class UnnamedAgent(Agent):

	def next_tactic(self, state, goal_id):
		pass
	def guidance(self, state):
		pass
```

