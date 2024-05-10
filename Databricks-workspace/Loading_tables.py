# Databricks notebook source
# Loading the data into dataframe
# Creating a spark session
from pyspark.sql import SparkSession

spark = (SparkSession
         .builder
         .appName("Table Loading")
         .getOrCreate())

sc = spark.sparkContext  

# COMMAND ----------

# Creating the [Posts] dataframe
# File location -- Recall our mount storage
file_location = "/mnt/capstoneFolders/ml_training/Posts/*"

posts = spark.read.parquet(file_location)
display(posts)

# COMMAND ----------

# Creating the schema for [posttypes] table
from pyspark.sql.types import *

PT_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("Type", StringType(), True)
])

# COMMAND ----------

# Creating the [posttypes] dataframe
file_location = "/mnt/capstoneFolders/ml_training/PostTypes.txt"

postType = (spark.read
  .option("header", "true") #to show the header as column
  .option("sep", ",") #to handle with csv file
  .schema(PT_schema) #structuring the table as we created for defining the schema
  .csv(file_location)) #file path

display(postType)

# COMMAND ----------

# Creating the schema for the [users] table
from pyspark.sql.types import *

users_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("Age", IntegerType(), True),
    StructField("CreationDate", DateType(), True),
    StructField("DisplayName", StringType(), True),
    StructField("DownVotes", IntegerType(), True),
    StructField("EmailHash", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Reputation", IntegerType(), True),
    StructField("UpVotes", IntegerType(), True),
    StructField("Views", IntegerType(), True),
    StructField("WebsiteUrl", StringType(), True),
    StructField("AccountId", IntegerType(), True)
])

# COMMAND ----------

# Creating the [users] dataframe
file_location = "/mnt/capstoneFolders/ml_training/users.csv"

users = (spark.read
  .option("header", "true") #to show the header as column
  .option("sep", ",") #to handle with csv file
  .schema(users_schema) #structuring the table as we created for defining the schema
  .csv(file_location)) #file path

display(users)

# COMMAND ----------

'''
Saving the dataframes for easy retrieval
Save the 3 tables to databricks local file system
'''
posts.write.parquet("/tmp/project/posts.parquet")
postType.write.parquet("/tmp/project/PostType.parquet")
users.write.parquet("/tmp/project/user.parquet")

# COMMAND ----------

# review the local file system
display(dbutils.fs.ls("/tmp/project/"))
