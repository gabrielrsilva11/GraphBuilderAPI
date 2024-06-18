from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "text-embedding-preview-0409",
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

sentence = [' ', 'found', 'Connacht', 'remained', 'has', 'a', 'part', 'of', 'the', 'province', 'of', 'Munster', 'ever', 'since', '.', ' ', 'This', 'hypothesis', 'correlation', 'disproven', 'has', 'been', 'by', 'evidence', 'showing', 'that', 'predicts', 'only', 'HIV', 'infection', ',', 'not', 'homosexuality', 'nor', 'recreational', 'drug', 'pharmaceutical', 'use', ',', 'who', 'develop', 'will', 'AIDS', '.', ' ', 'The', 'results', 'of', '1927', 'Championships', 'World', ',', 'where', 'won', 'Sonja', 'Henie', 'in', '3', 'decision', 'DASH', '2', '(', 'or', '7', '8', 'vs.', 'points', 'ordinal', ')', 'over', 'the', 'Champion', 'defending', 'Olympic', 'and', 'World', 'Herma', 'Szabo', 'of', 'Austria', ',', 'was', 'controversial', ',', 'as', 'were', 'all', 'three', 'of', 'five', 'judges', 'that', 'gave', 'Sonja', 'Henie', 'first', 'place', '-', 'ordinals', 'Norwegian', '(', 'points', '1', '+', '1', '+', '1', '+', '2', '+', '2', '=', '7', ')', 'while', 'received', 'Szabo', 'first', 'place', '-', 'ordinals', 'from', 'an', 'Austrian', 'and', 'a', 'Judge', 'German', '(', 'points', '1', '+', '1', '+', '2', '+', '2', '+', '2', '=', '8', ')', '.', ' ', 'In', 'argued', 'astrophysics', 'and', 'the', 'field', 'celestial', 'mechanics', 'of', 'physics', ',', 'Alhazen', ',', 'in', 'Alhazen', 'Epitome', 'of', 'Astronomy', ',', 'be', 'that', 'needed', 'Ptolemaic', 'models', 'to', 'understood', 'be', 'in', 'terms', 'of', 'physical', 'objects', 'rather', 'than', 'abstract', 'hypotheses', ';', 'in', 'other', 'words', 'that', 'it', 'should', 'possible', 'to', 'create', 'physical', 'models', 'where', 'collide', '(', 'for', 'example', ')', 'none', 'of', 'the', 'bodies', 'celestial', 'would', 'with', 'each', 'other', '.', ' ', 'Given', 'that', 'was', 'the', 'origin', 'of', 'humanity', 'almost', 'certainly', 'in', 'Africa', ',', 'postulate', 'several', 'theories', 'that', 'were', 'the', 'hominids', 'first', 'in', 'Europe', 'in', 'Andalusia', ',', 'having', 'passed', 'across', 'the', 'Strait', 'of', 'Gibraltar', ';', 'the', 'paintings', 'earliest', 'known', 'of', 'humanity', 'have', 'been', 'in', 'the', 'Caves', 'of', 'Nerja', ',', 'Málaga', '.']

print(embed_text(sentence))


# import vertexai
# from vertexai.language_models import TextGenerationModel
#
#
# def interview(
#     temperature: float,
#     project_id: str,
#     location: str,
# ) -> str:
#     """Ideation example with a Large Language Model"""
#
#     vertexai.init(project=project_id, location=location)
#     # TODO developer - override these parameters as needed:
#     parameters = {
#         "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
#         "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
#         "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
#         "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
#     }
#
#     model = TextGenerationModel.from_pretrained("text-bison@002")
#     response = model.predict(
#         """Frase: Hoje tomou posse o Presidente da República Federativa do Brasil.
# Chunks: Hoje, tomou posse, o palerma, o Presidente, da, República Federativa do Brasil
# Relações: Hoje tomou advmod;  tomou posse obj; tomou "o President" nsubj; "o Presidente" " República Federativa do Brasil" nmod
# Pseudo Labels:  Presidente da República Federativa do Brasil, CARGO
#
# Resenha: O nome dele é José Sarney.
# Chunks: o nome, dele, é, José, Sarney
# Relações: "O nome" é nsubj; José Sarney flat:name
# Pseudo Labels:  José Sarney, PESSOA
#
#
# Frase: Hoje tomou posse o Ferro Rodrigues como Presidente da Assembleia da República Portuguesa.
# Chunks: Hoje tomou posse, o palerma o Ferro Rodrigues, como, Presidente da Assembleia da República Portuguesa
# Relações: Hoje tomou advmod; tomou posse obj; "o palerma" "Ferro Rodrigues" nmod; Presidente "República Portuguesa" nmod
# Pseudo Labels:""",
#         **parameters,
#     )
#     print(f"Response from Model: {response.text}")
#
#     return response.text
#
#
# print(interview(temperature=0.5, project_id="stellar-zoo-405011", location="europe-west4"))


# import requests
# import google.auth
# import google.auth.transport.requests
# from google.oauth2 import service_account
#
# SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
# # Replace 'NAME_OF_FILE' with your service account JSON file name that you downloaded
# SERVICE_ACCOUNT_FILE = 'stellar-zoo-405011-0fe44fa19720.json'
# cred = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# # Create an authentication request
# auth_req = google.auth.transport.requests.Request()
#
# # Refresh the credentials
# cred.refresh(auth_req)
#
# # Obtain the bearer token
# bearer_token = cred.token
# # Define the base URL for your specific region (us-central1 in this example)
# full_url = "https://eu-west2-aiplatform.googleapis.com/v1beta1/projects/1011886731182/locations/europe-west2/models/llama2-13b-emanuel@1:predict"
#
# # Replace 'awesome-dogfish-399811' with your GCP project ID
# project_id = "stellar-zoo-405011"
#
# # Replace '639689267970310144' with the Endpoint ID from the model dashboard
# endpoint_id = "7569007259284406272"
#
# headers = {
#     "Authorization": "Bearer {bearer_token}".format(bearer_token=bearer_token),
#     "Content-Type": "application/json"
# }
#
# request_body = {
#     "instances": [
#         {
#             "prompt": "Write a poem about Valencia.",
#             "max_length": 200,
#             "top_k": 10
#         }
#     ]
# }
# #full_url = base_url.format(project_id=project_id, endpoint_id=endpoint_id)
#
# # Send a POST request to the model endpoint
# resp = requests.post(full_url, json=request_body, headers=headers)
#
# # Print the response from the model
# print(resp)

# from typing import Dict, List, Union
#
# from google.cloud import aiplatform
# from google.protobuf import json_format
# from google.protobuf.struct_pb2 import Value
#
#
# def predict_custom_trained_model_sample(
#     project: str,
#     endpoint_id: str,
#     instances: Union[Dict, List[Dict]],
#     location: str = "us-central1",
#     api_endpoint: str = "us-central1-aiplatform.googleapis.com",
# ):
#     """
#     `instances` can be either single instance of type dict or a list
#     of instances.
#     """
#     # The AI Platform services require regional API endpoints.
#     client_options = {"api_endpoint": api_endpoint}
#     # Initialize client that will be used to create and send requests.
#     # This client only needs to be created once, and can be reused for multiple requests.
#     client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
#     # The format of each instance should conform to the deployed model's prediction input schema.
#     instances = instances if isinstance(instances, list) else [instances]
#     instances = [
#         json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
#     ]
#     parameters_dict = {}
#     parameters = json_format.ParseDict(parameters_dict, Value())
#     endpoint = client.endpoint_path(
#         project=project, location=location, endpoint=endpoint_id
#     )
#     response = client.predict(
#         endpoint=endpoint, instances=instances, parameters=parameters
#     )
#     print("response")
#     print(" deployed_model_id:", response.deployed_model_id)
#     # The predictions are a google.protobuf.Value representation of the model's predictions.
#     predictions = response.predictions
#     for prediction in predictions:
#         print(" prediction:", dict(prediction))
#
# predict_custom_trained_model_sample(
#     project="1011886731182",
#     endpoint_id="3004609036944408576",
#     location="europe-west2",
#     instances={ "prompt": "Write a poem about portugal"}
# )