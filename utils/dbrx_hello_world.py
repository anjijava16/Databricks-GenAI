import getpass
import os

from utils import init

init()
# os.environ["DATABRICKS_HOST"] = "hhttps://dbc-46aea839-6b3d.cloud.databricks.com"
# if "DATABRICKS_TOKEN" not in os.environ:
#     os.environ["DATABRICKS_TOKEN"] = getpass.getpass(
#         "Enter your Databricks access token: "
#     )
from databricks_langchain import ChatDatabricks

chat_model = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    temperature=0.1,
    max_tokens=256,
    # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
)
response = chat_model.invoke("What is MLflow?")
print(f"response ={response}")

# You can also pass a list of messages
messages = [
    ("system", "You are a chatbot that can answer questions about Databricks."),
    ("user", "What is Databricks Model Serving?"),
]
response1 = chat_model.invoke(messages)
print(f"response1 ={response1}")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot that can answer questions about {topic}.",
        ),
        ("user", "{question}"),
    ]
)

chain = prompt | chat_model
response3 = chain.invoke(
    {
        "topic": "Databricks",
        "question": "What is Unity Catalog?",
    }
)
print(f" response3= {response3}")

print(" Stream output start here ")

for chunk in chat_model.stream("How are you?"):
    print(chunk.content, end="|")
