"""Test cases for the minif2f experiment."""

from pandas import DataFrame, Series
from pantograph import Server
from tqdm import tqdm
import pytest
from loguru import logger
from .utils import verify_theorem_loading


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

@pytest.mark.advance
def test_load_theorem(minif2f_server: Server, minif2f_test: DataFrame, minif2f_valid: DataFrame):
    """Comprehensive test for loading multiple theorems.
    use pytest -m "not advance" to skip this test.
    """
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
            minif2f_server.restart()
    # Test test theorems
    logger.info("Testing test theorems...")
    failed_test = []
    for i, theorem in tqdm(enumerate(minif2f_test.formal_statement), 
                          desc="Testing test theorems", 
                          total=len(minif2f_test)):
        is_valid, _ = verify_theorem_loading(minif2f_server, theorem)
        if not is_valid:
            failed_test.append(i)
            minif2f_server.restart()
    
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
    if failed_valid or failed_test:
        if failed_valid:
            err_msg = f"Failed valid theorems: {failed_valid}"
        if failed_test:
            err_msg += f"\nFailed test theorems: {failed_test}"
        raise AssertionError(err_msg)
