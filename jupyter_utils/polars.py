"""Utility functions for working with polars."""

import uuid
from typing import NamedTuple, ParamSpec, TypeVar

import numpy as np
import polars as pl

T = TypeVar("T", bound=tuple)
P = ParamSpec("P")


class TrainTestSplit(NamedTuple):
    """Tuple of the test and train split."""

    test: pl.LazyFrame
    train: pl.LazyFrame


def column_to_series(
    df: pl.LazyFrame | pl.DataFrame, column: int | str = 0
) -> pl.Series:
    """Get a column from a dataframe.

    Args:
        df (pl.LazyFrame | pl.DataFrame): The dataframe to retrieve the column from.
        column (int | str, optional): The column index or name. Defaults to 0.

    Returns:
        pl.Series: The series from the dataframe.
    """
    if isinstance(column, int):
        column = df.columns[column]
    return df.lazy().select(column).collect().to_series()


def lazy_height(df: pl.LazyFrame | pl.DataFrame) -> int:
    """Get the height of a data frame."""
    if isinstance(df, pl.DataFrame):
        return df.height
    return df.select(pl.len()).collect().item()


def train_test_split(
    df: pl.LazyFrame | pl.DataFrame,
    test_size: float | int = 0.25,
    train_size: float | int | None = None,
    seed: int | None = None,
    shuffle: bool = True,
) -> TrainTestSplit:
    """Create a train test split from a polars dataframe.

    Args:
        df (pl.LazyFrame | pl.DataFrame): The input data frame.
        test_size (float | int, optional): The size of the test set. Defaults to 0.25.
        train_size (float | int | None, optional): The size of the train set.
            Defaults to None, so compliments the test_size.
        seed (int | None, optional): The random seed. Ignored if shuffle is False.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Raises:
        ValueError: If the train_size and test_size are greater than the total number
            of samples.

    Returns:
        TrainTestSplit: A tuple of the train and test split data.
    """
    df = df.lazy()
    count = lazy_height(df)
    if isinstance(test_size, float):
        test_size = int(np.round(test_size * count))
    if train_size is None:
        train_size = count - test_size
    elif isinstance(train_size, float):
        train_size = int(np.round(train_size * count))
    if test_size + train_size > count:
        raise ValueError(
            "Test size and train size must not sum to more than the values."
            + f" {test_size} + {train_size} > {count} "
        )
    idx = np.arange(count)

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(idx)

    # Add temporary column for splitting data.
    col_name = "_id"
    while col_name in df.columns:
        col_name = f"_id{uuid.uuid4()}"

    df = df.with_columns(pl.Series(name=col_name, values=idx))
    test_df = df.filter(pl.col(col_name) < test_size)
    train_df = df.filter(
        pl.col(col_name).is_between(test_size, test_size + train_size, closed="left")
    )
    return TrainTestSplit(
        test=test_df.select(pl.exclude(col_name)),
        train=train_df.select(pl.exclude(col_name)),
    )
