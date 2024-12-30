import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#Embeddings endpoints convert text into a vector (array of float). Here is an example using GTEgte:
response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)
