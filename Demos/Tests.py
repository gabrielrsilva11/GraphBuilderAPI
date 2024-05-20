# from typing import List
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
#
#
# def embed_text(
#     texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
#     task: str = "RETRIEVAL_DOCUMENT",
#     model_name: str = "textembedding-gecko@003",
# ) -> List[List[float]]:
#     """Embeds texts with a pre-trained, foundational model."""
#     model = TextEmbeddingModel.from_pretrained(model_name)
#     inputs = [TextEmbeddingInput(text, task) for text in texts]
#     embeddings = model.get_embeddings(inputs)
#     return [embedding.values for embedding in embeddings]
#
# sentence = ['Parliament', 'officer', 's', 'press', 'told', 'Reuters', 'that', 'convene', 'Speaker', 'Savi', 'Toomas', 'will', 'an', 'college', 'electoral', 'involving', '101', 'MPs', 'and', 'and', '273', 'representatives', 'local', 'government', 'on', 'September', '20', '.']
#
# print(embed_text(sentence))


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


