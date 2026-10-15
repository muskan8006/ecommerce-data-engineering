from pyspark.sql import functions as F

from common import (
    create_spark_session,
    load_all_data,
    save_parquet,
    save_csv,
)


def analyze_customer_behavior(data):
    """
    Analyze customer orders and identify repeat customers.
    """

    orders = data["orders"]
    customers = data["customers"]

    customer_orders = (
        orders
        .join(
            customers.select(
                "customer_id",
                "customer_unique_id",
                "customer_city",
                "customer_state",
            ),
            on="customer_id",
            how="left",
        )
        .groupBy(
            "customer_unique_id",
            "customer_city",
            "customer_state",
        )
        .agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.min(
                "order_purchase_timestamp"
            ).alias("first_order_timestamp"),
            F.max(
                "order_purchase_timestamp"
            ).alias("last_order_timestamp"),
        )
        .withColumn(
            "customer_type",
            F.when(
                F.col("total_orders") > 1,
                "Repeat Customer",
            ).otherwise("One-Time Customer"),
        )
    )

    return customer_orders


def summarize_customer_behavior(customer_orders):
    """
    Summarize one-time and repeat customer counts.
    """

    customer_summary = (
        customer_orders
        .groupBy("customer_type")
        .agg(
            F.count("*").alias("customer_count"),
            F.sum("total_orders").alias("orders_placed"),
            F.round(
                F.avg("total_orders"),
                2,
            ).alias("average_orders_per_customer"),
        )
        .orderBy(F.desc("customer_count"))
    )

    return customer_summary


def analyze_category_reviews(data):
    """
    Calculate category-level review scores and sales metrics.
    Reviews are aggregated at order level before joining sales.
    """

    items = data["order_items"]
    products = data["products"]
    reviews = data["reviews"]
    translations = data["category_translation"]

    # Convert review scores safely to numeric values.
    # Invalid values become NULL instead of raising a cast error.
    reviews_clean = (
        reviews
        .withColumn(
            "review_score_numeric",
            F.expr("try_cast(trim(review_score) AS DOUBLE)"),
        )
        .withColumn(
            "review_score_numeric",
            F.when(
                F.col("review_score_numeric").between(1, 5),
                F.col("review_score_numeric"),
            ).otherwise(F.lit(None).cast("double")),
        )
    )

    # Aggregate reviews to one row per order.
    order_reviews = (
        reviews_clean
        .groupBy("order_id")
        .agg(
            F.avg(
                "review_score_numeric"
            ).alias("order_average_review_score"),
            F.count(
                "review_score_numeric"
            ).alias("review_count"),
        )
    )

    # Map each product to its category.
    product_categories = products.select(
        "product_id",
        "product_category_name",
    )

    item_categories = (
        items
        .join(
            product_categories,
            on="product_id",
            how="left",
        )
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

    # One record per order and category.
    order_category_sales = (
        item_categories
        .groupBy("order_id", "category_name")
        .agg(
            F.count("*").alias("items_in_category"),
            F.sum("price").alias("category_revenue"),
        )
    )

    # Attach order-level reviews without multiplying review rows.
    category_reviews = (
        order_category_sales
        .join(
            order_reviews,
            on="order_id",
            how="left",
        )
        .groupBy("category_name")
        .agg(
            F.countDistinct(
                "order_id"
            ).alias("sales_orders"),
            F.sum(
                "items_in_category"
            ).alias("items_sold"),
            F.sum(
                "category_revenue"
            ).alias("category_revenue"),
            F.round(
                F.avg("order_average_review_score"),
                2,
            ).alias("average_review_score"),
            F.count(
                "order_average_review_score"
            ).alias("orders_with_reviews"),
        )
        .orderBy(F.desc("category_revenue"))
    )

    return category_reviews


def main():
    spark = create_spark_session()

    try:
        data = load_all_data(spark)

        print("\n========== CUSTOMER BEHAVIOR ==========\n")

        customer_orders = analyze_customer_behavior(data)
        customer_orders.show(10, truncate=False)

        save_parquet(
            customer_orders,
            "customer_behavior",
        )

        print("\n========== CUSTOMER SUMMARY ==========\n")

        customer_summary = summarize_customer_behavior(
            customer_orders
        )
        customer_summary.show(truncate=False)

        save_csv(
            customer_summary,
            "customer_summary",
        )

        print("\n========== CATEGORY REVIEWS ==========\n")

        category_reviews = analyze_category_reviews(data)
        category_reviews.show(20, truncate=False)

        save_parquet(
            category_reviews,
            "category_reviews",
        )

        print("\nCustomer and product analytics completed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
