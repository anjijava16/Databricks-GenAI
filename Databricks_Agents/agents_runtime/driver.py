# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic langchain_databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "You will be provided with a document and asked a question about it.\n\nAnswer the question from the document.\nDocument:\nFarm Utility Exemption Notice\nAct 1441 of 2013 provides an exemption from state and local sales taxes for electricity,\nnatural gas, and liquefied petroleum gas used by qualifying agricultural structures and\nqualifying aquaculture and horticulture equipment beginning January 1, 2014.\nThe eligible utility must be separately metered and used only for the purpose of the\nexemption. If a utility is sold for any other purpose, it will not be eligible for the exemption.\nBefore the exemption is allowed, the farmer seeking the exemption must obtain a\ncertificate from DFA to provide to the utility supplier.\nQualifying agricultural structures are defined as\nA. A poultry or livestock facility used for commercial production, including without\nlimitation a broiler or turkey grow-out house, laying house, hatching unit,\nnursery unit, breeding house, farrowing unit, and feed-out house;\nB. A cattle or dairy facility, including without limitation a milking parlor, milk\ncollection unit, and refrigeration unit and\nC. A greenhouse used for commercial production.\nHorticulture means the initial production and cultivation of fruits, vegetables, tree nuts,\ntrees, shrubs, vines, and florists unless the cultivation of these items are at a retail or\nwholesale facility from which the items are sold.\nAquaculture is defined as the active cultivation of domesticated fish that are spawned,\ngrown, managed, harvested, and marketed on an annual, semiannual, biennial, or short\nterm basis in waters that are confined within a pond, tank, or lake that is situated entirely\non the premises of a single owner and that, except under abnormal flood conditions, are\nin no way connected by water or with any other flowing stream or body of water; or body\nof water not situated on the premises of the owner.\nQualifying aquaculture or horticulture equipment includes:\nA. A cooling unit, collection unit, or irrigation equipment used in a commercial\nhorticulture operation;\nB. Equipment used to pump and aerate a pond used in a commercial aquaculture\noperation; and\nC. A holding and sorting tank used in a commercial aquaculture operation.\nThe forms to obtain the necessary certificate to provide to the utility suppliers are\navailable by contacting our office at 501-682-7105, Sales and Use Tax Section, P O Box\n3566, Little Rock, AR 72203-3566 or are located in the Forms section on the Sales and\nUse Tax page of the DFA website www.dfa.arkansas.gov .\n\nQuestion: Is a greenhouse a qualifying agricultural structure?\n"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            "langchain==0.2.16",
            "langchain-community==0.2.16",
            "langgraph-checkpoint==1.0.12",
            "langgraph==0.2.16",
            "pydantic",
            "langchain_databricks", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "You will be provided with a document and asked a question about it.\n\nAnswer the question from the document.\nDocument:\nFarm Utility Exemption Notice\nAct 1441 of 2013 provides an exemption from state and local sales taxes for electricity,\nnatural gas, and liquefied petroleum gas used by qualifying agricultural structures and\nqualifying aquaculture and horticulture equipment beginning January 1, 2014.\nThe eligible utility must be separately metered and used only for the purpose of the\nexemption. If a utility is sold for any other purpose, it will not be eligible for the exemption.\nBefore the exemption is allowed, the farmer seeking the exemption must obtain a\ncertificate from DFA to provide to the utility supplier.\nQualifying agricultural structures are defined as\nA. A poultry or livestock facility used for commercial production, including without\nlimitation a broiler or turkey grow-out house, laying house, hatching unit,\nnursery unit, breeding house, farrowing unit, and feed-out house;\nB. A cattle or dairy facility, including without limitation a milking parlor, milk\ncollection unit, and refrigeration unit and\nC. A greenhouse used for commercial production.\nHorticulture means the initial production and cultivation of fruits, vegetables, tree nuts,\ntrees, shrubs, vines, and florists unless the cultivation of these items are at a retail or\nwholesale facility from which the items are sold.\nAquaculture is defined as the active cultivation of domesticated fish that are spawned,\ngrown, managed, harvested, and marketed on an annual, semiannual, biennial, or short\nterm basis in waters that are confined within a pond, tank, or lake that is situated entirely\non the premises of a single owner and that, except under abnormal flood conditions, are\nin no way connected by water or with any other flowing stream or body of water; or body\nof water not situated on the premises of the owner.\nQualifying aquaculture or horticulture equipment includes:\nA. A cooling unit, collection unit, or irrigation equipment used in a commercial\nhorticulture operation;\nB. Equipment used to pump and aerate a pond used in a commercial aquaculture\noperation; and\nC. A holding and sorting tank used in a commercial aquaculture operation.\nThe forms to obtain the necessary certificate to provide to the utility suppliers are\navailable by contacting our office at 501-682-7105, Sales and Use Tax Section, P O Box\n3566, Little Rock, AR 72203-3566 or are located in the Forms section on the Sales and\nUse Tax page of the DFA website www.dfa.arkansas.gov .\n\nQuestion: Is a greenhouse a qualifying agricultural structure?\n"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "main"
schema = "default"
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})