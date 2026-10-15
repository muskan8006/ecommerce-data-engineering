import os
import sys
import pyspark

# Force Spark workers to use the active virtual environment's Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

print("Python:", sys.executable)
print("PySpark:", pyspark.__version__)

spark = (
    SparkSession.builder
    .appName("WorkerTest")
    .master("local[1]")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .config("spark.python.worker.faulthandler.enabled", "true")
    .getOrCreate()
)

try:
    spark.range(5).show()

    df = spark.createDataFrame(
        [(1, "test"), (2, "working")],
        ["id", "status"]
    )
    df.show()

    print("PYTHON WORKER TEST PASSED")
finally:
    spark.stop()
