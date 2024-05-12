# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1 Spark preparation

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import SparkSession

# Creating Spark Session
spark = (SparkSession
         .builder
         .appName("ML Model")
         .getOrCreate())

sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2 Prepare a UDF (User Defined Function)
# MAGIC
# MAGIC Create UDF to embed the ML model trained in the `ML_training` notebook. This model will be used for Posts data sentiment analysis.

# COMMAND ----------

def predictions_udf(df, ml_model, stringindexer):
    from pyspark.sql.functions import col, regexp_replace, lower, trim
    from pyspark.ml import PipelineModel

    # Filter out empty body text
    df = df.filter("Body is not null")
    # Making sure the naming of the columns are consistent with the model
    df = df.select(col("Body").alias("text"), col("Tags"))
    # Preprocessing of the feature column
    cleaned = df.withColumn('text', regexp_replace('text', r"http\S+", "")) \
                    .withColumn('text', regexp_replace('text', r"[^a-zA-z]", " ")) \
                    .withColumn('text', regexp_replace('text', r"\s+", " ")) \
                    .withColumn('text', lower('text')) \
                    .withColumn('text', trim('text')) 

    # Load in the saved pipeline model
    model = PipelineModel.load(ml_model)

    # Making the prediction
    prediction = model.transform(df)

    predicted = prediction.select(col('text'), col('Tags'), col('prediction'))

    # Decoding the indexer
    from pyspark.ml.feature import StringIndexerModel, IndexToString

    # Load in the StringIndexer that was saved
    indexer = StringIndexerModel.load(stringindexer)

    # Initialize the IndexToString converter
    i2s = IndexToString(inputCol = 'prediction', outputCol = 'decoded', labels = indexer.labels)
    converted = i2s.transform(predicted)

    # Display the important columns
    return converted

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3 Load Posts files and ML model

# COMMAND ----------

posts = spark.read.parquet("/mnt/capstoneFolders/Posts/*")
ml_model = "/mnt/capstoneFolders/model"
stringindexer = "/mnt/capstoneFolders/stringindexer"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.4 Run model to do `Sentiment Analysis`

# COMMAND ----------

#Producing the sentiment analysis
result = predictions_udf(posts,ml_model, stringindexer)
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.5 Summarize which topics are the most popular

# COMMAND ----------

# change the column name 
topics = result.withColumnRenamed('decoded', 'topic').select('topic')

# Aggregate the topics and calculate the total qty of each topic
topic_qty = topics.groupBy(col("topic")).agg(count('topic').alias('qty')).orderBy(desc('qty'))
topic_qty.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.6 Save the result file to the `BI` folder
# MAGIC
# MAGIC Since spark is a distribution system, if you don't anything, the file you saved will be a folder with a couple of files. 
# MAGIC In order to save a single file, we need a function to move the csv file move out of the folder, rename it, and delete the folder. leave the single csv file alone.

# COMMAND ----------

def crt_sgl_file(result_path):
        # write the result to a folder container several files
        path = "/mnt/capstoneFolders/BI/ml_result"
        topic_qty.write.option("delimiter", ",").option("header", "true").mode("overwrite").csv(path)

        # list the folder, find the csv file 
        filenames = dbutils.fs.ls(path)
        name = ''
        for filename in filenames:
            if filename.name.endswith('csv'):
                org_name = filename.name

        # copy the csv file to the path you want to save, in this example, we use  "/mnt/deBDProject/BI/ml_result.csv"
        dbutils.fs.cp(path + '/'+ org_name, result_path)

        # delete the folder
        dbutils.fs.rm(path, True)

        print('single file created')

# COMMAND ----------

# run the function
result_path = "/mnt/capstoneFolders/BI/ml_result.csv"

crt_sgl_file(result_path)
