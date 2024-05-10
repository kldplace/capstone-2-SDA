# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Join tables and filter data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1 Prepare necessary libraries and load data
# MAGIC

# COMMAND ----------

# Import necessary libraries and functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower

# COMMAND ----------

# Creating Spark Session
spark = (SparkSession
         .builder
         .appName("ML Model")
         .getOrCreate())

sc = spark.sparkContext

# COMMAND ----------

# Read in the tables
posts = spark.read.parquet("/tmp/project/posts.parquet")
postType = spark.read.parquet("/tmp/project/PostType.parquet")
Users = spark.read.parquet("/tmp/project/user.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2 Join the tables Posts and postTypes by it post type id

# COMMAND ----------

# at this moment, we only use [Posts] and [posttypes] to train the model.

# join [Posts] and [posttypes] with the [posttypes] id
df = posts.join(postType, posts.PostTypeId == postType.id)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3 Filter the data

# COMMAND ----------

# MAGIC %md
# MAGIC In the ***posttypes*** table, there is a column called `Type` which indicates if the posts is a question or an answer. We only need the 'question' entires. For these 'Question' rows, we will run machine learning model on the join the 'Body' column of the 'Posts' table. To tell what topic this post is about.

# COMMAND ----------

# Filter the dataframe to only include questions
df = df.filter(col("Type") == "Question")
display(df)

# COMMAND ----------

# Formatting the 'Body' and `Tag` columns for machine learning training
df = (df.withColumn('Body', regexp_replace(df.Body, r'<.*?>', '')) # Transforming HTML code to strings
      .withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " ")) # Making a list of the tags
)

display(df)

# COMMAND ----------

# Filter the dataframe to only include questions
df = df.filter(col("Type") == "Question")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.4 Create a checkpoint to save the dataframe to file only contain the `Body` and `Tag` we need. 

# COMMAND ----------

df = df.select(col("Body").alias("text"), col("Tags"))

# COMMAND ----------

# Producing the tags as individual tags instead of an array
# This is duplicating the posts for each possible tag
df = df.select("text", explode("Tags").alias("tags"))
display(df)

# COMMAND ----------

# saving the file as a checkpoint (in case the cluster gets terminated)
df.write.parquet("/tmp/project.df.parquet")

# COMMAND ----------

'''
Saving the dataframe to memory for repetitive use.
caching allows Spark to avoid recomputing the dataset each time it's accessed.
'''
df.cache()
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Based on the above dataframe, prepare data from machine learning

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1. Text Cleaning Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC `pyspark.sql.functions.regexp_replace` is used to process the text
# MAGIC
# MAGIC 1. Remove URLs such as `http://stackoverflow.com`
# MAGIC 2. Remove special characters
# MAGIC 3. Substituting multiple spaces with single space
# MAGIC 4. Lowercase all text
# MAGIC 5. Trim the leading/trailing whitespaces

# COMMAND ----------

# Preprocessing the data
cleaned = df.withColumn('text', regexp_replace('text', r"http\S+", "")) \
                    .withColumn('text', regexp_replace('text', r"[^a-zA-z]", " ")) \
                    .withColumn('text', regexp_replace('text', r"\s+", " ")) \
                    .withColumn('text', lower('text')) \
                    .withColumn('text', trim('text')) 
display(cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Machine Learning Model Training
# MAGIC ##### 3.1 Feature Transformer

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 3.1.1 Tokenizer
# MAGIC <hr>
# MAGIC
# MAGIC In machine learning, a ***tokenizer*** is a tool used to break down text or sequences of characters into smaller units, such as words, phrases, or individual characters. This process is essential for natural language processing (NLP) tasks like text classification, sentiment analysis, and language translation. Tokenization enables algorithms to understand and process textual data by converting it into a format that can be analyzed and modeled effectively.

# COMMAND ----------

from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
tokenized = tokenizer.transform(cleaned)

display(tokenized)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 3.1.2 Stopword Removal
# MAGIC <hr>
# MAGIC
# MAGIC In machine learning, ***stop words removal*** is the process of filtering out common words that occur frequently in a language but typically do not carry significant meaning or contribute to the understanding of text. Examples of stop words include "the," "is," "and," "in," etc. Removing these words helps to improve the efficiency and accuracy of natural language processing (NLP) tasks such as text classification, sentiment analysis, and topic modeling by focusing on the words that are more informative and relevant to the analysis.

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
stopword = stopword_remover.transform(tokenized)

display(stopword)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 3.1.3 CountVectorizer (TF - Term Frequency)
# MAGIC <hr>
# MAGIC
# MAGIC
# MAGIC In machine learning, ***CountVectorizer (Term Frequency)*** is a technique used to convert a collection of text documents into a numerical feature representation. It counts the frequency of each word (or term) in the documents and creates a matrix where each row represents a document and each column represents a unique word, with the cell values indicating the frequency of each word in the corresponding document. This method is useful for various text-based machine learning tasks such as document classification, clustering, and information retrieval.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
cv_model = cv.fit(stopword)
text_cv = cv_model.transform(stopword)

display(text_cv)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 3.1.4 TF-IDF Vectorization
# MAGIC <hr>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ***TF-IDF (Term Frequency-Inverse Document Frequency)*** vectorization is a technique used in machine learning to convert a collection of text documents into numerical feature vectors. It combines the concepts of term frequency (TF) and inverse document frequency (IDF).
# MAGIC TF-IDF vectorization computes a numerical value for each term in each document, which represents both its local importance (how often it appears in the document) and its global importance (how much information it provides across all documents). This allows machine learning algorithms to effectively process and analyze textual data for tasks like document classification, information retrieval, and clustering.

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF

idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
idf_model = idf.fit(text_cv)
text_idf = idf_model.transform(text_cv)

display(text_idf)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.2 Label Encoding
# MAGIC <hr>
# MAGIC <hr>
# MAGIC
# MAGIC ***Label encoding*** is a technique in machine learning used to convert categorical data into numerical form. It assigns a unique integer to each category or class in the categorical variable. This transformation is useful for algorithms that require numerical input, such as regression and neural networks. However, it's important to note that label encoding introduces an ordinal relationship between the categories, which may not always be appropriate for all algorithms, especially those that assume no inherent order among categories.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

label_encoder = StringIndexer(inputCol = "tags", outputCol = "label")
le_model = label_encoder.fit(text_idf)
final = le_model.transform(text_idf)

display(final)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.4 Model Training

# COMMAND ----------

from pyspark.ml.classification import 

lr = LogisticRegression(maxIter=100)
lr_model = lr.fit(final)
predictions = lr_model.transform(final)

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.5 Model Evalution
# MAGIC <hr>
# MAGIC <hr>
# MAGIC
# MAGIC
# MAGIC ***Model evaluation*** in machine learning refers to the process of assessing how well a trained model performs on unseen data. It involves comparing the predictions made by the model with the actual ground truth labels or values to measure its accuracy,  Model evaluation helps determine the effectiveness and generalization ability of the model, ensuring that it can make reliable predictions on new data.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
roc_auc = evaluator.evaluate(predictions)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.6 Create a Pipeline

# COMMAND ----------

# Importing all the libraries
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Preparing the data
# Step 1: Creating the joined table
df = posts.join(postType, posts.PostTypeId == postType.id)
# Step 2: Selecting only Question posts
df = df.filter(col("Type") == "Question")
# Step 3: Formatting the raw data
df = (df.withColumn('Body', regexp_replace(df.Body, r'<.*?>', ''))
      .withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " "))
)
# Step 4: Selecting the columns
df = df.select(col("Body").alias("text"), col("Tags"))
# Step 5: Getting the tags
df = df.select("text", explode("Tags").alias("tags"))
# Step 6: Clean the text
cleaned = df.withColumn('text', regexp_replace('text', r"http\S+", "")) \
                    .withColumn('text', regexp_replace('text', r"[^a-zA-z]", " ")) \
                    .withColumn('text', regexp_replace('text', r"\s+", " ")) \
                    .withColumn('text', lower('text')) \
                    .withColumn('text', trim('text')) 

# Machine Learning
# Step 1: Train Test Split
train, test = cleaned.randomSplit([0.9, 0.1], seed=20200819)
# Step 2: Initializing the transfomers
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5)
label_encoder = StringIndexer(inputCol = "tags", outputCol = "label")
lr = LogisticRegression(maxIter=100)
# Step 3: Creating the pipeline
pipeline = Pipeline(stages=[tokenizer, stopword_remover, cv, idf, label_encoder, lr])
# Step 4: Fitting and transforming (predicting) using the pipeline
pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Save the Model file to Azure storage

# COMMAND ----------

# Saving model object.
pipeline_model.save('/mnt/capstoneFolders/model')

# Saving the String Indexer to decode the encoding. We need it in the future Sentiment Analysis.
le_model.save('/mnt/capstoneFolders/stringindexer')

# COMMAND ----------

# Review the directory
display(dbutils.fs.ls("/mnt/capstoneFolders/model"))
