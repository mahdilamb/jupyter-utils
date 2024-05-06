"""Fixtures for pytest."""

import contextlib
import os

import pytest


@pytest.fixture(scope="session")
def create_working_directory_mocker():
    """Fixture for creating a working directory mocker."""
    start_dir = os.getcwd()

    @contextlib.contextmanager
    def mock(dir: str):
        try:
            os.chdir(dir)
            yield
        finally:
            os.chdir(start_dir)

    return mock
