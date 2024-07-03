from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-3131138635619485.5.azuredatabricks.net/serving-endpoints"
)

embeddings = client.embeddings.create(
  input='Your string for the embedding model goes here',
  model="databricks-gte-large-en"
)

print(embeddings.data[0].embedding)
