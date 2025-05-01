from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST')
# Alternatively in a Databricks notebook you can use this:
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(DATABRICKS_TOKEN)

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-46aea839-6b3d.cloud.databricks.com/serving-endpoints"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        {
            "role": "user",
            "content": "Tell me about Large Language Models"
        }
    ],
    model="databricks-meta-llama-3-70b-instruct",
    max_tokens=256
)

print(chat_completion.choices[0].message.content)
