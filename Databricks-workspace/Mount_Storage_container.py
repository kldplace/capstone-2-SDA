# Databricks notebook source
# Import [dotenv] package
%pip install python-dotenv

# COMMAND ----------

from dotenv import load_dotenv
import os

load_dotenv()

# My environment variables
storage_account_access_key = os.environ.get('STORAGE_ACCOUNT_ACCESS_KEY')
storage_account_name = os.environ.get('STORAGE_ACCOUNT_NAME')
container_name = os.environ.get('CONTAINER_NAME')

# COMMAND ----------

# Set up the configuration with the storage account name and access key
configs = {
  f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key
}

# Mount the Blob Storage container to DBFS
dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
  mount_point = "/mnt/capstoneFolders",
  extra_configs = configs)

# COMMAND ----------

# Use this to unmount the Blob Storage container to DBFS
#dbutils.fs.unmount("/mnt/capstoneFolders")
