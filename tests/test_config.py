import os

import pytest


@pytest.fixture
def setup():
    os.makedirs("src", exist_ok=True)
    os.makedirs("tests", exist_ok=True)
    return None


def test_setup(setup):
    assert os.path.exists("src")
