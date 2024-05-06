"""Fixtures for pytest."""

import contextlib
import os
from collections.abc import Generator

import polars as pl
import pytest


@pytest.fixture(scope="session")
def answer_df() -> Generator[pl.DataFrame, None, None]:
    """Fixture for a dataframe with a single column 'answer' containing numbers."""
    df = pl.DataFrame((pl.int_range(end=42, eager=True).alias("answer"),))
    yield df


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
