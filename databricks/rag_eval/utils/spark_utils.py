from typing import Union

import pandas as pd


def normalize_spark_df(
    df: Union[
        pd.DataFrame,
        "pyspark.sql.DataFrame",  # noqa: F821
        "pyspark.sql.connect.dataframe.DataFrame",  # noqa: F821
    ],
) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    # Import pyspark here to avoid hard dependency on pyspark
    import pyspark.sql.connect.dataframe

    if isinstance(df, (pyspark.sql.DataFrame, pyspark.sql.connect.dataframe.DataFrame)):
        return df.toPandas()
    return df
