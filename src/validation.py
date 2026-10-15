from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from common import create_spark_session, load_all_data, save_csv


def record_check(report, check_name, passed, details=""):
    """Add one validation result to the report."""
    report.append({
        "check_name": check_name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })


def validate_data(spark: SparkSession, data: dict) -> DataFrame:
    """Validate the loaded e-commerce datasets."""

    report = []

    required_tables = [
        "orders",
        "customers",
        "order_items",
        "products",
        "payments",
        "reviews",
        "sellers",
        "geolocation",
        "category_translation",
    ]

    # 1. Check required tables and whether each table has rows.
    for table_name in required_tables:
        if table_name not in data:
            record_check(
                report,
                f"{table_name}_exists",
                False,
                "Required table is missing"
            )
            continue

        df = data[table_name]

        record_check(
            report,
            f"{table_name}_exists",
            True,
            "Table is available"
        )

        row_count = df.count()
        record_check(
            report,
            f"{table_name}_not_empty",
            row_count > 0,
            f"Rows found: {row_count}"
        )

    # 2. Check unique identifiers.
    unique_key_checks = {
        "orders": ["order_id"],
        "customers": ["customer_id"],
        "products": ["product_id"],
        "sellers": ["seller_id"],
    }

    for table_name, columns in unique_key_checks.items():
        if table_name not in data:
            continue

        df = data[table_name]
        missing_columns = [
            col for col in columns if col not in df.columns
        ]

        if missing_columns:
            record_check(
                report,
                f"{table_name}_unique_key",
                False,
                f"Missing columns: {missing_columns}"
            )
            continue

        null_condition = None
        for col in columns:
            condition = F.col(col).isNull()
            null_condition = (
                condition
                if null_condition is None
                else null_condition | condition
            )

        null_count = df.filter(null_condition).count()
        duplicate_count = (
            df.groupBy(*columns)
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        passed = null_count == 0 and duplicate_count == 0

        record_check(
            report,
            f"{table_name}_unique_key",
            passed,
            f"Null keys: {null_count}; duplicate keys: {duplicate_count}"
        )

    # 3. Check important order columns.
    if "orders" in data:
        orders = data["orders"]

        required_order_columns = [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ]

        missing = [
            col for col in required_order_columns
            if col not in orders.columns
        ]

        record_check(
            report,
            "orders_required_columns",
            len(missing) == 0,
            f"Missing columns: {missing}" if missing else "All present"
        )

        if "order_id" in orders.columns:
            null_order_ids = orders.filter(
                F.col("order_id").isNull()
            ).count()

            record_check(
                report,
                "orders_order_id_not_null",
                null_order_ids == 0,
                f"Null order IDs: {null_order_ids}"
            )

    # 4. Validate order item prices and freight values.
    if "order_items" in data:
        items = data["order_items"]

        for col_name in ["price", "freight_value"]:
            if col_name not in items.columns:
                record_check(
                    report,
                    f"order_items_{col_name}_valid",
                    False,
                    f"Column {col_name} is missing"
                )
                continue

            numeric_col = F.expr(
                f"try_cast(trim(`{col_name}`) AS DOUBLE)"
            )

            invalid_count = items.filter(
                numeric_col.isNull() |
                F.isnan(numeric_col) |
                (numeric_col < 0)
            ).count()

            record_check(
                report,
                f"order_items_{col_name}_valid",
                invalid_count == 0,
                f"Invalid or negative values: {invalid_count}"
            )

    # 5. Safely validate review scores (malformed values won't crash).
    if "reviews" in data:
        reviews = data["reviews"]

        if "review_score" not in reviews.columns:
            record_check(
                report,
                "review_score_valid",
                False,
                "review_score column is missing"
            )
        else:
            score = F.expr(
                "try_cast(trim(`review_score`) AS DOUBLE)"
            )

            invalid_count = reviews.filter(
                score.isNull() |
                F.isnan(score) |
                (score < 1) |
                (score > 5)
            ).count()

            record_check(
                report,
                "review_score_valid",
                invalid_count == 0,
                f"Invalid review scores: {invalid_count}"
            )

    # 6. Check order/customer relationship.
    if "orders" in data and "customers" in data:
        orders = data["orders"]
        customers = data["customers"]

        if (
            "customer_id" in orders.columns
            and "customer_id" in customers.columns
        ):
            order_customers = orders.select("customer_id").distinct()
            known_customers = customers.select("customer_id").distinct()

            unmatched = order_customers.join(
                known_customers,
                on="customer_id",
                how="left_anti"
            ).count()

            record_check(
                report,
                "orders_customer_reference",
                unmatched == 0,
                f"Customer IDs without a matching customer: {unmatched}"
            )

    # 7. Check order item relationships.
    if "order_items" in data and "orders" in data:
        items = data["order_items"]
        orders = data["orders"]

        if "order_id" in items.columns and "order_id" in orders.columns:
            unmatched = (
                items.select("order_id").distinct()
                .join(
                    orders.select("order_id").distinct(),
                    on="order_id",
                    how="left_anti"
                )
                .count()
            )

            record_check(
                report,
                "order_items_order_reference",
                unmatched == 0,
                f"Order IDs without a matching order: {unmatched}"
            )

    # 8. Create the validation report.
    report_df = spark.createDataFrame(report)

    print("\n========== VALIDATION SUMMARY ==========")
    report_df.show(truncate=False)

    failed_count = sum(
        1 for item in report if item["status"] == "FAIL"
    )
    print(f"Total checks: {len(report)}")
    print(f"Failed checks: {failed_count}")

    return report_df


def run_validation(data: dict) -> DataFrame:
    """
    Pipeline-compatible entry point.
    pipeline.py calls run_validation(data), so Spark is obtained
    from the DataFrames already loaded in the data dictionary.
    """

    if not data:
        raise ValueError("The data dictionary is empty.")

    first_df = next(
        (value for value in data.values()
         if isinstance(value, DataFrame)),
        None
    )

    if first_df is None:
        raise ValueError(
            "No Spark DataFrames were found in the data dictionary."
        )

    spark = first_df.sparkSession
    return validate_data(spark, data)


def main():
    """Allow validation.py to run independently."""

    spark = create_spark_session()
    try:
        data = load_all_data(spark)
        report_df = run_validation(data)
        save_csv(report_df, "validation_report")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
