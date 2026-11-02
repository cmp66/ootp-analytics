from pandas import DataFrame, concat


def numeric_columns(df: DataFrame) -> DataFrame:
    """The numeric columns of a ratings frame (the index, normally the player ID, is kept).

    The stat calculations only use numeric ratings. Every string column is a block of its own in pandas 3, so
    merging the whole ratings frame (names, dates, flags) leaves a fragmented frame and a PerformanceWarning for
    every column added afterwards. The copy joins the numeric columns into a few blocks."""
    return df.select_dtypes(include="number").copy()  # copy() consolidates the blocks


def add_constant_columns(df: DataFrame, values: dict) -> DataFrame:
    """df with one extra column per entry of values (each filled with that value), added in one step.

    DataFrame.assign adds the columns one at a time; with 50 league values on top of 100 stat columns that
    fragments the frame. Like assign, an existing column of the same name is replaced."""
    extra = DataFrame({name: [value] * len(df) for name, value in values.items()}, index=df.index)
    return concat([df.drop(columns=[c for c in values if c in df.columns]), extra], axis=1)


def consolidate(df: DataFrame) -> DataFrame:
    """A copy of df whose columns are joined into a few blocks (string columns become object columns).

    pandas 3 keeps every string column in a block of its own, so a csv with 100 text columns is fragmented as
    soon as it is read and every merge or insert on it raises a PerformanceWarning."""
    strings = [c for c, dtype in df.dtypes.items() if str(dtype) == "str"]
    return df.astype({c: "object" for c in strings}).copy()
