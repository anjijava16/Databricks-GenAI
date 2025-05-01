import getpass
import os

from utils import init

init()

from databricks_langchain import ChatDatabricks


#llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
        },
    }
]

# supported tool_choice values: "auto", "required", "none", function name in string format,
# or a dictionary as {"type": "function", "function": {"name": <<tool_name>>}}
model = llm.bind_tools(tools, tool_choice="auto")

messages = [{"role": "user", "content": "What is the current temperature of Chicago?"}]
print(model.invoke(messages))