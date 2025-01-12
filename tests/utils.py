from pantograph import Server
from pantograph.expr import GoalState
from loguru import logger

def apply_tactics(server:Server, state:GoalState, tactics:list):
    states = [state]
    logger.opt(raw=True).debug(f"{state}\n")
    for tactic in tactics:
        logger.opt(raw=True).debug(f"{tactic}\n")
        state = server.goal_tactic(state, 0, tactic)
        if not state.is_solved:
            logger.opt(raw=True).debug(f"{state}\n")
        else:
            logger.debug("close proof")
        states.append(state)
    return states

default_header = """
open BigOperators Real Nat Topology Rat
"""

def verify_theorem_loading(server: Server, theorem: str) -> tuple[bool, str]:
    """Helper function to verify theorem loading."""
    try:
        # the first argument is for the `open` command
        unit = server.load_sorry(f"{default_header}\n{theorem} := by sorry")[1]
        goal_state, message = unit.goal_state, '\n'.join(unit.messages)
        is_valid = (
            goal_state is not None and 
            len(goal_state.goals) == 1 and 
            'error' not in message.lower()
        )
        return is_valid, message
    except Exception as e:
        logger.error(f"Exception while loading theorem: {e}")
        return False, str(e)