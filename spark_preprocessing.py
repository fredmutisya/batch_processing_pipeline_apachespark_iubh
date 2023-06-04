import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum, rank, from_unixtime, month, dayofmonth, year
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, DateType
import pandas as pd

# Initialize the spark session
spark = SparkSession.builder.appName('eCommerce')\
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")\
    .getOrCreate()


from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, DateType

# Define the expected schema 
schema = StructType([
    StructField("item_id", IntegerType()),
    StructField("status", StringType()),
    StructField("created_at", DateType()),
    StructField("sku", StringType()),
    StructField("price", FloatType()),
    StructField("qty_ordered", IntegerType()),
    StructField("grand_total", FloatType()),
    StructField("increment_id", StringType()),
    StructField("category_name_1", StringType()),
    StructField("sales_commission_code", StringType()),
    StructField("discount_amount", FloatType()),
    StructField("payment_method", StringType()),
    StructField("Working Date", DateType()),
    StructField("BI Status", StringType()),
    StructField(" MV ", StringType()),
    StructField("Year", IntegerType()),
    StructField("Month", IntegerType()),
    StructField("Customer Since", DateType()),
    StructField("M-Y", StringType()),
    StructField("FY", StringType()),
    StructField("Customer ID", IntegerType())
])

# Load the pyspark dataframe as df_ecommerce with the updated schema
df_ecommerce = spark.read.option('header', 'true').csv('ecommerce.csv', schema=schema)


# Show the df_ecommerce
df_ecommerce.show()

# Transform into proper data types
df_ecommerce = df_ecommerce.withColumn("price", df_ecommerce["price"].cast(FloatType()))
df_ecommerce = df_ecommerce.withColumn("qty_ordered", df_ecommerce["qty_ordered"].cast(IntegerType()))
df_ecommerce = df_ecommerce.withColumn("grand_total", df_ecommerce["grand_total"].cast(FloatType()))
df_ecommerce = df_ecommerce.withColumn("discount_amount", df_ecommerce["discount_amount"].cast(FloatType()))
df_ecommerce = df_ecommerce.withColumn("Year", df_ecommerce["Year"].cast(IntegerType()))
df_ecommerce = df_ecommerce.withColumn("Month", df_ecommerce["Month"].cast(IntegerType()))
df_ecommerce = df_ecommerce.withColumn("Customer ID", df_ecommerce["Customer ID"].cast(IntegerType()))

df_ecommerce = df_ecommerce.withColumn("created_at", F.to_date(F.col("created_at"), "MM/dd/yyyy"))
df_ecommerce = df_ecommerce.withColumn("Working Date", F.to_date(F.col("Working Date"), "MM/dd/yyyy"))
df_ecommerce = df_ecommerce.withColumn("Customer Since", F.to_date(F.col("Customer Since"), "MM/dd/yyyy"))

# Breaking 'created_at' into year, month, and day
df_ecommerce = df_ecommerce.withColumn("created_at_year", year("created_at"))
df_ecommerce = df_ecommerce.withColumn("created_at_month", month("created_at"))
df_ecommerce = df_ecommerce.withColumn("created_at_day", dayofmonth("created_at"))



# Filter out rows with invalid year values
df_ecommerce = df_ecommerce.filter((df_ecommerce["Year"] >= 1900) & (df_ecommerce["Year"] <= 2100))


# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df_ecommerce.toPandas()

# Write the df_ecommerce dataframe to a CSV file
output_file = 'ml_output.csv'
pandas_df.to_csv(output_file, header=True, index=False, mode='w')

