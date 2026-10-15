
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    IntegerType,
)

from common import (
    create_spark_session,
    load_all_data,
    save_csv,
)


def profile_dataset(data):
    """Profile tables for row counts, nulls and duplicate keys."""

    if not data:
        raise ValueError("No datasets were loaded.")

    primary_keys = {
        "orders": ["order_id"],
        "order_items": ["order_id", "order_item_id"],
        "payments": ["order_id", "payment_sequential"],
        "customers": ["customer_id"],
        "products": ["product_id"],
        "sellers": ["seller_id"],
        "reviews": ["review_id"],
    }

    results = []

    for table_name, df in data.items():
        print(f"\nProfiling table: {table_name}")

        # Count rows in the table
        row_count = df.count()

        # Calculate null values in one aggregation instead
        # of launching a separate count job for every column.
        null_expressions = [
            F.sum(
                F.when(F.col(column).isNull(), 1).otherwise(0)
            ).alias(f"null_{index}")
            for index, column in enumerate(df.columns)
        ]

        null_result = df.agg(*null_expressions).first()

        null_count = sum(
            int(value or 0) for value in null_result
        )

        # Count duplicate primary-key groups
        duplicate_count = 0

        if table_name in primary_keys:
            keys = primary_keys[table_name]

            duplicate_count = (
                df.groupBy(*keys)
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

        results.append(
            (
                table_name,
                int(row_count),
                int(len(df.columns)),
                int(null_count),
                int(duplicate_count),
            )
        )

        print(
            f"{table_name}: "
            f"rows={row_count}, "
            f"columns={len(df.columns)}, "
            f"null_values={null_count}, "
            f"duplicate_key_groups={duplicate_count}"
        )

    print("\nDEBUG: Profiling results collected:")

    for result in results:
        print(repr(result))

    # Explicit schema avoids automatic type inference
    summary_schema = StructType([
        StructField("table_name", StringType(), False),
        StructField("row_count", LongType(), False),
        StructField("column_count", IntegerType(), False),
        StructField("total_null_values", LongType(), False),
        StructField("duplicate_key_groups", LongType(), False),
    ])

    summary = data[next(iter(data))].sparkSession.createDataFrame(
        results,
        schema=summary_schema,
    )

    print("\nDEBUG: Summary DataFrame created. Testing show()...")
    summary.show(truncate=False)

    save_csv(summary, "data_profiling_summary")

    print("Data profiling summary saved.")

    return summary


def check_foreign_keys(data):
    """Check for IDs that do not exist in their parent table."""

    if not data:
        raise ValueError("No datasets were loaded.")

    checks = [
        ("order_items", "order_id", "orders", "order_id"),
        ("payments", "order_id", "orders", "order_id"),
        ("orders", "customer_id", "customers", "customer_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("order_items", "seller_id", "sellers", "seller_id"),
    ]

    results = []

    for child_table, child_col, parent_table, parent_col in checks:
        print(
            f"\nChecking foreign key: "
            f"{child_table}.{child_col} -> "
            f"{parent_table}.{parent_col}"
        )

        child = (
            data[child_table]
            .select(F.col(child_col).alias("child_id"))
            .filter(F.col(child_col).isNotNull())
            .distinct()
        )

        parent = (
            data[parent_table]
            .select(F.col(parent_col).alias("parent_id"))
            .filter(F.col(parent_col).isNotNull())
            .distinct()
        )

        unmatched_count = (
            child.join(
                parent,
                child["child_id"] == parent["parent_id"],
                "left_anti",
            )
            .count()
        )

        print(f"Unmatched IDs: {unmatched_count}")

        results.append(
            (
                child_table,
                child_col,
                parent_table,
                parent_col,
                int(unmatched_count),
            )
        )

    report_schema = StructType([
        StructField("child_table", StringType(), False),
        StructField("child_column", StringType(), False),
        StructField("parent_table", StringType(), False),
        StructField("parent_column", StringType(), False),
        StructField("unmatched_id_count", LongType(), False),
    ])

    spark = data[next(iter(data))].sparkSession

    report = spark.createDataFrame(
        results,
        schema=report_schema,
    )

    report.show(truncate=False)

    save_csv(report, "foreign_key_checks")

    print("Foreign-key report saved.")

    return report


def main():
    spark = create_spark_session()

    try:
        print("\n========== LOADING DATA ==========")
        data = load_all_data(spark)

        print("\n========== DATA PROFILING ==========")
        profile_dataset(data)

        print("\n========== FOREIGN KEY CHECKS ==========")
        check_foreign_keys(data)

        print("\nProfiling completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
