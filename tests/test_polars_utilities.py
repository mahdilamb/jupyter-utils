"""Tests for polars utility functions."""

import polars as pl
import pytest

import magicbox.polars as polars_utils


@pytest.mark.parametrize(
    ("count", "test_size", "train_size", "expected_heights"),
    [
        (10, 0.2, 0.4, (2, 4, 4)),
        (10, 0.2, None, (2, 8, 0)),
        (11, 0.2, None, (2, 9, 0)),
        (11, 0.2, 0.9, None),
    ],
)
def test_split_sizes_are_correct(
    count: int,
    test_size: float | int,
    train_size: float | int | None,
    expected_heights: tuple[int, int] | None,
):
    """Test that test/train split works for polars.

    Args:
        count (int): The number of samples to create.
        test_size (float | int): The test_size.
        train_size (float | int | None): The train_size.
        expected_heights (tuple[int, int]): The expected heights.
            If None, expect and exception.
    """
    df = pl.DataFrame((pl.int_range(end=count, eager=True).alias("answer"),))

    if expected_heights is None:
        with pytest.raises(ValueError):
            polars_utils.train_test_split(
                df, test_size=test_size, train_size=train_size, seed=42
            )
    else:
        actual_heights = tuple(
            _.height
            for _ in pl.collect_all(
                polars_utils.train_test_split(
                    df, test_size=test_size, train_size=train_size, seed=42
                )
            )
        )
        assert actual_heights == expected_heights, (
            f"Expected splitting {count} into (test={test_size}, train={train_size}) "
            + f"to be {expected_heights}. Got {actual_heights}."
        )


@pytest.mark.parametrize(
    ("column", "transform"),
    [
        (0, lambda _: _),
        ("answer", lambda _: _),
        (0, pl.DataFrame.lazy),
        ("answer", pl.DataFrame.lazy),
    ],
)
def test_getting_column_from_dataframe(
    answer_df: pl.DataFrame, column: int | str, transform
):
    """Test that getting a column works.

    Args:
        answer_df (pl.DataFrame): The input dataframe.
        column (int | str): the column to retrieve
        transform (Callable[[pl.DataFrame], pl.LazyFrame|pl.DataFrame]):
            Function called to transform the input dataframe.
    """
    answer_df = transform(answer_df)
    assert (
        polars_utils.column_to_series(answer_df, column).name == "answer"
    ), "Expected the 0th column to be answer."


@pytest.mark.parametrize(
    ("df",), [(pl.DataFrame(),), (pl.LazyFrame(),), (pl.DataFrame().lazy(),)]
)
def test_no_throw_for_height_of_empty_dataframe(df):
    """Test that empty dataframes return a 0 and don't thrown an error."""
    assert polars_utils.lazy_height(df) == 0


@pytest.mark.parametrize(
    ("df", "expected", "checker"),
    [
        (
            pl.DataFrame({"a": [None, 0, 1]}),
            pl.DataFrame(
                {"column": ["a"], "inf": [0], "nan": [0], "null": [1], "len": [3]}
            ),
            polars_utils.check_numeric_columns,
        ),
        (
            pl.DataFrame({"a": [float("nan"), 0, 1]}),
            pl.DataFrame(
                {"column": ["a"], "inf": [0], "nan": [1], "null": [0], "len": [3]}
            ),
            polars_utils.check_numeric_columns,
        ),
        (
            pl.DataFrame({"a": [float("inf"), 0, 1]}),
            pl.DataFrame(
                {"column": ["a"], "inf": [1], "nan": [0], "null": [0], "len": [3]}
            ),
            polars_utils.check_numeric_columns,
        ),
        (
            pl.DataFrame({"a": [float("-inf"), 0, 1]}),
            pl.DataFrame(
                {"column": ["a"], "inf": [1], "nan": [0], "null": [0], "len": [3]}
            ),
            polars_utils.check_numeric_columns,
        ),
        (
            pl.DataFrame({"a": [float("-nan"), 0, 1]}),
            pl.DataFrame(
                {"column": ["a"], "inf": [0], "nan": [1], "null": [0], "len": [3]}
            ),
            polars_utils.check_numeric_columns,
        ),
        (
            pl.DataFrame({"a": ["Hey :D-<", "", None]}),
            pl.DataFrame({"column": ["a"], "null": [1], "empty": [1], "len": [3]}),
            polars_utils.check_string_columns,
        ),
        (
            pl.DataFrame(
                {"a": ["Hello", "world", None]},
                schema={"a": pl.Enum(("Hello", "world"))},
            ),
            pl.DataFrame({"column": ["a"], "null": [1], "empty": [0], "len": [3]}),
            polars_utils.check_string_columns,
        ),
    ],
)
def test_checkers(df: pl.DataFrame, expected, checker):
    """Test the various checkers."""
    actual: pl.DataFrame = checker(df)
    assert (actual == expected).sum_horizontal().item() == len(
        actual.columns
    ), "Expected the check to return valid metrics."


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
