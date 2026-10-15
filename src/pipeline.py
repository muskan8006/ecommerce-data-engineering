from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common import (
    create_spark_session,
    load_all_data,
    save_parquet,
    save_csv,
)

from profiling import profile_dataset, check_foreign_keys
from revenue import (
    build_order_analytics,
    calculate_monthly_revenue,
    calculate_category_revenue,
    calculate_monthly_category_revenue,
)
from delivery import (
    calculate_delivery_metrics,
    calculate_state_performance,
    build_delivery_summary,
)
from customer_product import (
    analyze_customer_behavior,
    summarize_customer_behavior,
    analyze_category_reviews,
)
from spark_sql import (
    register_temp_views,
    run_revenue_sql,
    run_delivery_sql,
    check_category_results,
)
from validation import run_validation


def main():
    spark = create_spark_session()

    try:
        print("\n========== 1. LOADING DATA ==========\n")
        data = load_all_data(spark)

        print("\n========== 2. DATA PROFILING ==========\n")
        profile_dataset(data)
        check_foreign_keys(data)

        print("\n========== 3. REVENUE ANALYTICS ==========\n")
        order_analytics = build_order_analytics(data)
        monthly_revenue = calculate_monthly_revenue(
            order_analytics
        )
        category_revenue = calculate_category_revenue(data)
        monthly_category = calculate_monthly_category_revenue(
            data
        )

        save_parquet(order_analytics, "order_analytics")
        save_csv(monthly_revenue, "monthly_revenue")
        save_csv(category_revenue, "category_revenue")

        # Rank categories separately within each month
        month_window = (
            Window
            .partitionBy("purchase_month")
            .orderBy(F.desc("category_revenue"))
        )

        monthly_category_ranked = monthly_category.withColumn(
            "category_rank",
            F.dense_rank().over(month_window),
        )

        save_parquet(
            monthly_category_ranked,
            "monthly_category_revenue",
        )

        print("\n========== 4. DELIVERY ANALYTICS ==========\n")
        delivery_data = calculate_delivery_metrics(data)
        state_performance = calculate_state_performance(
            delivery_data
        )
        delivery_summary = build_delivery_summary(
            delivery_data
        )

        save_parquet(delivery_data, "delivery_analytics")
        save_csv(
            state_performance,
            "state_delivery_performance",
        )
        save_csv(delivery_summary, "delivery_summary")

        print("\n========== 5. CUSTOMER ANALYTICS ==========\n")
        customer_data = analyze_customer_behavior(data)
        customer_summary = summarize_customer_behavior(
            customer_data
        )

        save_parquet(customer_data, "customer_behavior")
        save_csv(customer_summary, "customer_summary")

        print("\n========== 6. PRODUCT REVIEW ANALYTICS ==========\n")
        category_reviews = analyze_category_reviews(data)

        save_parquet(category_reviews, "category_reviews")

        print("\n========== 7. SPARK SQL ANALYTICS ==========\n")
        register_temp_views(spark, data)

        top_months, top_categories = run_revenue_sql(spark)
        sql_state_performance = run_delivery_sql(spark)

        save_csv(top_months, "sql_top_revenue_months")
        save_csv(top_categories, "sql_top_categories")
        save_csv(
            sql_state_performance,
            "sql_state_late_delivery",
        )

        check_category_results(
            spark,
            category_revenue,
            top_categories,
        )

        print("\n========== 8. VALIDATION ==========\n")
        validation_report = run_validation(data)

        validation_report.show(truncate=False)

        failed_checks = validation_report.filter(
            F.col("status") == "FAIL"
        ).count()

        print("\n========== PIPELINE SUMMARY ==========\n")
        print(f"Validation checks failed: {failed_checks}")

        if failed_checks == 0:
            print("Pipeline completed; all validation checks passed.")
        else:
            print(
                "Pipeline completed with validation failures. "
                "Review the validation report before submission."
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()