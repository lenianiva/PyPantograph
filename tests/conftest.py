import pytest
import pandas as pd
from pathlib import Path
from pantograph import Server
from loguru import logger
logger.remove()
LOGGER_FORMAT = "<green>{level}</green> | <lvl>{message}</lvl>"
logger.add(lambda msg: print(msg, end=''), format=LOGGER_FORMAT, colorize=True)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Experiment for MiniF2F
MINIF2F_VALID_FILE = PROJECT_ROOT / 'experiments/minif2f/valid.jsonl'
MINIF2F_TEST_FILE = PROJECT_ROOT / 'experiments/minif2f/test.jsonl'
MINIF2F_ROOT = PROJECT_ROOT / 'experiments/minif2f/MiniF2F'

@pytest.fixture
def minif2f_root():
    return MINIF2F_ROOT

@pytest.fixture
def minif2f_valid():
    return pd.read_json(MINIF2F_VALID_FILE, lines=True)

@pytest.fixture
def minif2f_test():
    return pd.read_json(MINIF2F_TEST_FILE, lines=True)

@pytest.fixture
def minif2f_server():
    server = Server(project_path=MINIF2F_ROOT, imports=['Mathlib', 'Aesop'])
    return server