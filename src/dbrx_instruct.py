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
  model="databricks-dbrx-instruct",
  max_tokens=256
)

print(chat_completion.choices[0].message.content)


#########
SELECT ai_query('databricks-dbrx-instruct',
    request => "<Please provide your input string here!>")

=====
curl \
  -u token:$DATABRICKS_TOKEN \
  -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  https://adb-3131138635619485.5.azuredatabricks.net/serving-endpoints/databricks-dbrx-instruct/invocations
