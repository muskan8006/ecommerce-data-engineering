from pyspark.sql import functions as F
from common import (
    create_spark_session,
    load_all_data,
    save_parquet,
    save_csv,
)

from revenue import (
    calculate_monthly_revenue,
    calculate_category_revenue,
)

from delivery import (
    calculate_delivery_metrics,
    calculate_state_performance,
)


def register_temp_views(spark, data):
    """
    Register source DataFrames as temporary SQL views.
    """

    for table_name, df in data.items():
        df.createOrReplaceTempView(table_name)

    print("Temporary SQL views registered.")


def run_revenue_sql(spark):
    """
    Run SQL queries for monthly and category revenue.
    """

    print("\n========== TOP 5 REVENUE MONTHS ==========\n")

    top_months = spark.sql(
        """
        SELECT
            DATE_FORMAT(
                TO_TIMESTAMP(order_purchase_timestamp),
                'yyyy-MM'
            ) AS purchase_month,
            ROUND(SUM(price), 2) AS monthly_revenue,
            COUNT(DISTINCT order_id) AS order_count
        FROM (
            SELECT
                oi.order_id,
                oi.price,
                o.order_purchase_timestamp
            FROM order_items oi
            INNER JOIN orders o
                ON oi.order_id = o.order_id
        ) item_orders
        GROUP BY
            DATE_FORMAT(
                TO_TIMESTAMP(order_purchase_timestamp),
                'yyyy-MM'
            )
        ORDER BY monthly_revenue DESC
        LIMIT 5
        """
    )

    top_months.show(truncate=False)

    print("\n========== TOP 10 CATEGORIES ==========\n")

    top_categories = spark.sql(
        """
        SELECT
            COALESCE(
                t.product_category_name_english,
                p.product_category_name,
                'unknown'
            ) AS category_name,
            ROUND(SUM(oi.price), 2) AS category_revenue,
            COUNT(*) AS item_count,
            COUNT(DISTINCT oi.order_id) AS order_count
        FROM order_items oi
        LEFT JOIN products p
            ON oi.product_id = p.product_id
        LEFT JOIN category_translation t
            ON p.product_category_name =
               t.product_category_name
        GROUP BY
            COALESCE(
                t.product_category_name_english,
                p.product_category_name,
                'unknown'
            )
        ORDER BY category_revenue DESC
        LIMIT 10
        """
    )

    top_categories.show(truncate=False)

    return top_months, top_categories


def run_delivery_sql(spark):
    """
    Calculate late-delivery percentages by state using SQL.
    """

    print("\n========== LATE DELIVERY BY STATE ==========\n")

    state_late_delivery = spark.sql(
        """
        WITH parsed_orders AS (
            SELECT
                o.order_id,
                c.customer_state,
                o.order_status,
                TO_TIMESTAMP(
                    o.order_purchase_timestamp
                ) AS purchase_timestamp,
                TO_TIMESTAMP(
                    o.order_delivered_customer_date
                ) AS delivered_timestamp,
                TO_TIMESTAMP(
                    o.order_estimated_delivery_date
                ) AS estimated_timestamp
            FROM orders o
            LEFT JOIN customers c
                ON o.customer_id = c.customer_id
        ),
        valid_deliveries AS (
            SELECT
                *,
                (
                    CAST(delivered_timestamp AS LONG)
                    - CAST(purchase_timestamp AS LONG)
                ) / 86400.0 AS delivery_days,
                CASE
                    WHEN delivered_timestamp >
                         estimated_timestamp
                    THEN 1
                    ELSE 0
                END AS is_late
            FROM parsed_orders
            WHERE order_status = 'delivered'
              AND purchase_timestamp IS NOT NULL
              AND delivered_timestamp IS NOT NULL
              AND estimated_timestamp IS NOT NULL
              AND delivered_timestamp >= purchase_timestamp
        )
        SELECT
            customer_state,
            COUNT(DISTINCT order_id) AS delivered_orders,
            ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
            SUM(is_late) AS late_orders,
            ROUND(
                100.0 * SUM(is_late)
                / COUNT(DISTINCT order_id),
                2
            ) AS late_delivery_percentage
        FROM valid_deliveries
        GROUP BY customer_state
        ORDER BY late_delivery_percentage DESC
        """
    )

    state_late_delivery.show(30, truncate=False)

    return state_late_delivery


def check_category_results(
    spark,
    dataframe_categories,
    sql_categories,
):
    """
    Compare top category names produced by SQL and DataFrames.
    """

    dataframe_top = [
        row["category_name"]
        for row in (
            dataframe_categories
            .select("category_name")
            .limit(10)
            .collect()
        )
    ]

    sql_top = [
        row["category_name"]
        for row in (
            sql_categories
            .select("category_name")
            .collect()
        )
    ]

    print("\n========== SQL / DATAFRAME CHECK ==========\n")

    if dataframe_top == sql_top:
        print("PASS: Top 10 categories match.")
    else:
        print("WARNING: Top 10 category lists differ.")
        print("DataFrame:", dataframe_top)
        print("SQL:", sql_top)

    return dataframe_top == sql_top


def main():
    spark = create_spark_session()

    try:
        data = load_all_data(spark)

        register_temp_views(spark, data)

        top_months, top_categories = run_revenue_sql(spark)

        state_late_delivery = run_delivery_sql(spark)

        save_csv(top_months, "sql_top_revenue_months")
        save_csv(top_categories, "sql_top_categories")
        save_csv(
            state_late_delivery,
            "sql_state_late_delivery",
        )

        # Compare SQL and DataFrame category revenue results
        dataframe_categories = calculate_category_revenue(data)

        category_results_match = check_category_results(
            spark,
            dataframe_categories,
            top_categories,
        )

        if not category_results_match:
            print(
                "Review category ordering and revenue "
                "definitions before submitting."
            )

        print("\nSpark SQL analytics completed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
