from pyspark.sql import SparkSession
from config import MONGO_URI
import os

#python_path = r"C:\\Users\\USER\\Desktop\\Fuudiy\\Fuudiy-Backend\\fenv\\Scripts\\python.exe"
#python_path = r"D:\OneDrive - TOBB Ekonomi ve Teknoloji Üniversitesi\Masaüstü\Fuudiy Back\venv\Scripts\python.exe"
#python_path = r"/Users/berfinozcubuk/Desktop/tobb/24_25/fuudiy/.venv/bin/python"
#python_path = "/usr/bin/python3.9"
import sys
python_path = sys.executable

os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
def get_spark_session():
    """Create and return a Spark session configured for MongoDB."""
    spark = SparkSession.builder \
        .appName("MongoDB-Spark") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.optimizer.maxIterations", 1000) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .getOrCreate()
    return spark
spark = get_spark_session()
