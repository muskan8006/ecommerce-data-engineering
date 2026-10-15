from pyspark.sql import functions as F

from common import (
    create_spark_session,
    load_all_data,
    save_parquet,
    save_csv,
)


def build_order_analytics(data):
    """
    Build one row per order.

    Revenue is calculated from order item prices.
    Payment totals are calculated separately to avoid
    multiplying values when joining item and payment tables.
    """

    orders = data["orders"]
    items = data["order_items"]
    payments = data["payments"]
    customers = data["customers"]

    # Aggregate item values to order level first
    item_totals = (
        items
        .groupBy("order_id")
        .agg(
            F.sum("price").alias("item_revenue"),
            F.sum("freight_value").alias("freight_revenue"),
            F.count("*").alias("item_count"),
        )
    )

    # Aggregate payments separately
    payment_totals = (
        payments
        .groupBy("order_id")
        .agg(
            F.sum("payment_value").alias("total_payment"),
            F.count("*").alias("payment_count"),
        )
    )

    # Select the required customer columns
    customer_info = customers.select(
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    )

    # Join aggregated tables without multiplying order rows
    order_analytics = (
        orders
        .join(item_totals, on="order_id", how="left")
        .join(payment_totals, on="order_id", how="left")
        .join(customer_info, on="customer_id", how="left")
        .withColumn(
            "item_revenue",
            F.coalesce(F.col("item_revenue"), F.lit(0.0)),
        )
        .withColumn(
            "freight_revenue",
            F.coalesce(F.col("freight_revenue"), F.lit(0.0)),
        )
        .withColumn(
            "total_payment",
            F.coalesce(F.col("total_payment"), F.lit(0.0)),
        )
        .withColumn(
            "item_count",
            F.coalesce(F.col("item_count"), F.lit(0)),
        )
        .withColumn(
            "payment_count",
            F.coalesce(F.col("payment_count"), F.lit(0)),
        )
        .withColumn(
            "order_purchase_timestamp",
            F.to_timestamp("order_purchase_timestamp"),
        )
        .withColumn(
            "purchase_month",
            F.date_format(
                "order_purchase_timestamp",
                "yyyy-MM",
            ),
        )
    )

    return order_analytics


def calculate_monthly_revenue(order_analytics):
    """
    Calculate monthly item revenue and freight.
    """

    monthly_revenue = (
        order_analytics
        .groupBy("purchase_month")
        .agg(
            F.sum("item_revenue").alias("monthly_revenue"),
            F.sum("freight_revenue").alias("monthly_freight"),
            F.countDistinct("order_id").alias("order_count"),
        )
        .orderBy("purchase_month")
    )

    return monthly_revenue


def calculate_category_revenue(data):
    """
    Calculate revenue by product category.

    Each order item contributes its price to its category.
    """

    items = data["order_items"]
    products = data["products"]
    translations = data["category_translation"]

    # Attach product category to each item
    product_categories = products.select(
        "product_id",
        "product_category_name",
    )

    category_data = (
        items
        .join(product_categories, on="product_id", how="left")
        .join(
            translations,
            on="product_category_name",
            how="left",
        )
        .withColumn(
            "category_name",
            F.coalesce(
                F.col("product_category_name_english"),
                F.col("product_category_name"),
                F.lit("unknown"),
            ),
        )
    )

    category_revenue = (
        category_data
        .groupBy("category_name")
        .agg(
            F.sum("price").alias("category_revenue"),
            F.sum("freight_value").alias("category_freight"),
            F.count("*").alias("item_count"),
            F.countDistinct("order_id").alias("order_count"),
        )
        .orderBy(
            F.desc("category_revenue"),
        )
    )

    return category_revenue


def calculate_monthly_category_revenue(data):
    """
    Calculate revenue by month and product category.
    """

    items = data["order_items"]
    orders = data["orders"]
    products = data["products"]
    translations = data["category_translation"]

    # Extract the purchase month from orders
    order_months = (
        orders
        .select(
            "order_id",
            F.date_format(
                F.to_timestamp("order_purchase_timestamp"),
                "yyyy-MM",
            ).alias("purchase_month"),
        )
    )

    product_categories = products.select(
        "product_id",
        "product_category_name",
    )

    monthly_category_data = (
        items
        .join(order_months, on="order_id", how="inner")
        .join(product_categories, on="product_id", how="left")
        .join(
            translations,
            on="product_category_name",
            how="left",
        )
        .withColumn(
            "category_name",
            F.coalesce(
                F.col("product_category_name_english"),
                F.col("product_category_name"),
                F.lit("unknown"),
            ),
        )
    )

    monthly_category_revenue = (
        monthly_category_data
        .groupBy("purchase_month", "category_name")
        .agg(
            F.sum("price").alias("category_revenue"),
            F.countDistinct("order_id").alias("order_count"),
        )
        .orderBy("purchase_month", F.desc("category_revenue"))
    )

    return monthly_category_revenue


def main():
    spark = create_spark_session()

    try:
        data = load_all_data(spark)

        print("\n========== ORDER ANALYTICS ==========\n")

        order_analytics = build_order_analytics(data)
        order_analytics.show(10, truncate=False)

        save_parquet(
            order_analytics,
            "order_analytics",
        )

        print("\n========== MONTHLY REVENUE ==========\n")

        monthly_revenue = calculate_monthly_revenue(
            order_analytics
        )

        monthly_revenue.show(20, truncate=False)

        save_csv(
            monthly_revenue,
            "monthly_revenue",
        )

        print("\n========== CATEGORY REVENUE ==========\n")

        category_revenue = calculate_category_revenue(data)
        category_revenue.show(10, truncate=False)

        save_csv(
            category_revenue,
            "category_revenue",
        )

        print("\n========== MONTHLY CATEGORY REVENUE ==========\n")

        monthly_category_revenue = (
            calculate_monthly_category_revenue(data)
        )

        monthly_category_revenue.show(20, truncate=False)

        save_parquet(
            monthly_category_revenue,
            "monthly_category_revenue",
        )

        print("\n========== REVENUE SUMMARY ==========\n")

        order_analytics.agg(
            F.sum("item_revenue").alias("total_item_revenue"),
            F.sum("freight_revenue").alias("total_freight"),
            F.sum("total_payment").alias("total_payments"),
            F.countDistinct("order_id").alias("total_orders"),
        ).show(truncate=False)

        print("\nRevenue analytics completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
