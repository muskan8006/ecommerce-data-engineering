from pyspark.sql import functions as F

from common import (
    create_spark_session,
    load_all_data,
    save_parquet,
    save_csv,
)


def calculate_delivery_metrics(data):
    """
    Calculate delivery duration and late delivery status.
    """

    orders = data["orders"]
    customers = data["customers"]

    # Keep the required order columns
    order_columns = orders.select(
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    )

    # Add customer state information
    order_data = order_columns.join(
        customers.select(
            "customer_id",
            "customer_state",
        ).join(
            orders.select("order_id", "customer_id"),
            on="customer_id",
            how="inner",
        ).select("order_id", "customer_state"),
        on="order_id",
        how="left",
    )

    # Convert date columns to timestamps
    order_data = (
        order_data
        .withColumn(
            "purchase_timestamp",
            F.to_timestamp("order_purchase_timestamp"),
        )
        .withColumn(
            "delivered_timestamp",
            F.to_timestamp("order_delivered_customer_date"),
        )
        .withColumn(
            "estimated_timestamp",
            F.to_timestamp("order_estimated_delivery_date"),
        )
    )

    # Calculate delivery duration in hours
    order_data = order_data.withColumn(
        "delivery_hours",
        (
            F.col("delivered_timestamp").cast("long")
            - F.col("purchase_timestamp").cast("long")
        ) / 3600.0,
    )

    order_data = order_data.withColumn(
        "delivery_days",
        F.col("delivery_hours") / 24.0,
    )

    # A valid delivery requires delivered status and valid dates
    valid_condition = (
        (F.col("order_status") == "delivered")
        & F.col("purchase_timestamp").isNotNull()
        & F.col("delivered_timestamp").isNotNull()
        & F.col("estimated_timestamp").isNotNull()
        & (F.col("delivery_hours") >= 0)
    )

    order_data = order_data.withColumn(
        "is_valid_delivery",
        F.coalesce(valid_condition, F.lit(False)),
    )

    # Mark deliveries completed after the estimated date
    order_data = order_data.withColumn(
        "is_late",
        F.when(
            F.col("is_valid_delivery"),
            F.col("delivered_timestamp")
            > F.col("estimated_timestamp"),
        ).otherwise(F.lit(None).cast("boolean")),
    )

    return order_data


def calculate_state_performance(delivery_data):
    """
    Calculate delivery metrics grouped by customer state.
    """

    valid_deliveries = delivery_data.filter(
        F.col("is_valid_delivery")
    )

    state_performance = (
        valid_deliveries
        .groupBy("customer_state")
        .agg(
            F.countDistinct("order_id").alias("delivered_orders"),
            F.round(
                F.avg("delivery_days"),
                2,
            ).alias("avg_delivery_days"),
            F.round(
                F.expr(
                    "percentile_approx(delivery_days, 0.5)"
                ),
                2,
            ).alias("median_delivery_days"),
            F.sum(
                F.when(F.col("is_late"), 1).otherwise(0)
            ).alias("late_orders"),
        )
        .withColumn(
            "late_delivery_percentage",
            F.round(
                F.col("late_orders")
                / F.col("delivered_orders")
                * 100,
                2,
            ),
        )
        .orderBy(
            F.desc("late_delivery_percentage")
        )
    )

    return state_performance


def build_delivery_summary(delivery_data):
    """
    Create an overall delivery summary.
    """

    summary = delivery_data.agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.countDistinct(
            F.when(
                F.col("is_valid_delivery"),
                F.col("order_id"),
            )
        ).alias("valid_delivered_orders"),
        F.countDistinct(
            F.when(
                ~F.col("is_valid_delivery"),
                F.col("order_id"),
            )
        ).alias("excluded_orders"),
        F.round(
            F.avg(
                F.when(
                    F.col("is_valid_delivery"),
                    F.col("delivery_days"),
                )
            ),
            2,
        ).alias("average_delivery_days"),
        F.sum(
            F.when(F.col("is_late") == True, 1).otherwise(0)
        ).alias("late_orders"),
    )

    return summary


def main():
    spark = create_spark_session()

    try:
        data = load_all_data(spark)

        print("\n========== DELIVERY METRICS ==========\n")

        delivery_data = calculate_delivery_metrics(data)

        delivery_data.select(
            "order_id",
            "customer_state",
            "order_status",
            "delivery_days",
            "is_valid_delivery",
            "is_late",
        ).show(10, truncate=False)

        save_parquet(
            delivery_data,
            "delivery_analytics",
        )

        print("\n========== STATE PERFORMANCE ==========\n")

        state_performance = calculate_state_performance(
            delivery_data
        )

        state_performance.show(30, truncate=False)

        save_csv(
            state_performance,
            "state_delivery_performance",
        )

        print("\n========== DELIVERY SUMMARY ==========\n")

        summary = build_delivery_summary(delivery_data)
        summary.show(truncate=False)

        save_csv(
            summary,
            "delivery_summary",
        )

        print("\nDelivery analytics completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
