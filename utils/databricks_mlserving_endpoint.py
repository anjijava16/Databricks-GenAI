from openai import OpenAI
import os

DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://dbc-849408bf-e64a.cloud.databricks.com/serving-endpoints"
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
     },
  ],
  model="databricks-dbrx-instruct",
  max_tokens=256
)

print(chat_completion.choices[0].message.content)
