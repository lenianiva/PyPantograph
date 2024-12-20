import pandas as pd
from pandas import DataFrame, Series
from pantograph import Server
from tqdm import tqdm
import pytest
from loguru import logger

default_header = """
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
"""

def verify_theorem_loading(server: Server, theorem: str) -> tuple[bool, str]:
    """Helper function to verify theorem loading."""
    try:
        unit = server.load_sorry(f"{default_header}\n{theorem}")[2]
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

@pytest.mark.basic
def test_single_case(minif2f_server: Server, minif2f_test: DataFrame):
    """Test loading of a single theorem case."""
    logger.info("Starting single case test")
    test_theorem = minif2f_test.iloc[0].formal_statement
    is_valid, message = verify_theorem_loading(minif2f_server, test_theorem)
    if is_valid:
        logger.success("Single case test passed successfully")
    else:
        logger.error(f"Single case test failed with message: {message}")
    assert is_valid, f"Failed to load theorem: {message}"

@pytest.mark.basic
def test_load_theorem(minif2f_server: Server, minif2f_test: DataFrame, minif2f_valid: DataFrame):
    """Comprehensive test for loading multiple theorems."""
    logger.info("Theorem loading test")
    # Test valid theorems
    logger.info("Testing valid theorems...")
    failed_valid = []
    for i, theorem in tqdm(enumerate(minif2f_valid.formal_statement), 
                          desc="Testing valid theorems", 
                          total=len(minif2f_valid)):
        is_valid, _ = verify_theorem_loading(minif2f_server, theorem)
        if not is_valid:
            failed_valid.append(i)
    # Test test theorems
    logger.info("Testing test theorems...")
    failed_test = []
    for i, theorem in tqdm(enumerate(minif2f_test.formal_statement), 
                          desc="Testing test theorems", 
                          total=len(minif2f_test)):
        is_valid, _ = verify_theorem_loading(minif2f_server, theorem)
        if not is_valid:
            failed_test.append(i)
    
    # Report results
    total_valid = len(minif2f_valid)
    total_test = len(minif2f_test)
    failed_valid_count = len(failed_valid)
    failed_test_count = len(failed_test)
    logger.info(f"""
    Test Summary:
    Valid theorems: {total_valid - failed_valid_count}/{total_valid} passed
    Test theorems: {total_test - failed_test_count}/{total_test} passed
    """)
    # Detailed failure report
    if failed_valid:
        logger.error(f"Failed valid theorems: {failed_valid}")
    if failed_test:
        logger.error(f"Failed test theorems: {failed_test}")
    assert not failed_valid, f"{failed_valid_count} valid theorems failed"
    assert not failed_test, f"{failed_test_count} test theorems failed"

@pytest.mark.advance
def test_advance_cases():
    pass