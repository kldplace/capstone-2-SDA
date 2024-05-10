# Databricks notebook source
!pip install python-dotenv


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext dotenv
# MAGIC %dotenv

# COMMAND ----------

# Define the storage account access key
storage_account_access_key = ""

# Set up the configuration with your storage account name and access key
configs = {
  "fs.azure.account.key.capstone2datalake.blob.core.windows.net": storage_account_access_key
}

# Mount the Blob Storage container to DBFS
dbutils.fs.mount(
  source = "wasbs://capstone-2@capstone2datalake.blob.core.windows.net/",
  mount_point = "/mnt/capstoneFolders",
  extra_configs = configs)
