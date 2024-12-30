from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("llm_prompt_template")), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

#Let's try our prompt:
answer = (prompt | model | StrOutputParser()).invoke({'question':'How to start a Databricks cluster?', 'context': ''})
display_txt_as_html(answer)
