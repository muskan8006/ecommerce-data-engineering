import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def create_spark_session():
    """Create a Spark session using the current Python interpreter."""

    python_path = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

    spark = (
        SparkSession.builder
        .appName("EcommerceGauntlet")
        .master("local[2]")
        .config("spark.pyspark.python", python_path)
        .config("spark.pyspark.driver.python", python_path)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )

    return spark


def load_all_data(spark):
    """Load the Olist CSV datasets into Spark DataFrames."""

    filenames = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }

    data = {}

    for table_name, filename in filenames.items():
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {file_path}\n"
                "Check that the CSV file exists inside the data folder."
            )

        data[table_name] = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(file_path))
        )

        print(
            f"Loaded {table_name}: "
            f"{data[table_name].count()} rows"
        )

    return data



def save_parquet(df: DataFrame, name: str):
    """Save a manageable analytics DataFrame as a Parquet file using pandas."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}.parquet"

    pandas_df = df.toPandas()
    pandas_df.to_parquet(output_path, index=False)

    print(f"Saved Parquet: {output_path}")




def save_csv(df: DataFrame, name: str):
    """Save a manageable analytics DataFrame as a CSV file using pandas."""

    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}.csv"

    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)

    print(f"Saved CSV: {output_path}")

def assert_unique(df: DataFrame, columns, table_name: str):
    """Raise an error if the specified columns contain duplicate keys."""

    duplicate_rows = (
        df.groupBy(*columns)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )

    if duplicate_rows:
        raise AssertionError(
            f"Duplicate key found in table '{table_name}' "
            f"for columns: {columns}"
        )

    print(
        f"Unique key check passed: "
        f"{table_name} -> {columns}"
    )
