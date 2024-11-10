import yaml
from GraphConverterNetworkX import BuildNetworkx
from tqdm import tqdm
from rdflib import URIRef, Literal
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from typing import List
import pandas as pd
def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "textembedding-gecko@003",
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]


config_file = open('Benchmarks/Configs/dataset_builder.yaml', 'r')
config_data = yaml.load(config_file, Loader=yaml.FullLoader)
base_uri = config_data['graph_name']
connect_string = config_data['connection_uri']

ntx = BuildNetworkx(base_uri, base_uri, connect_string)
ids_to_fetch = [1, 2]
word_dict = {}
subject = ''
for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
    idx = ids_to_fetch[i]
    graph = ntx.fetch_graph([idx])
    words = []
    for s, p, o in graph.triples((None, URIRef(base_uri + "word"), None)):
        sentence_dict = {}
        for s2, p2, o2 in graph.triples((s, None, None)):
            subject = s2.__str__().split("#")[-1]
            predicate = p2.__str__().split("#")[-1]
            object = o2.__str__().split("#")[-1]
            word_dict[predicate] = object
        sentence_dict[subject] = word_dict
    words.append(sentence_dict)

df = pd.DataFrame.from_records(words)
df.to_excel(config_data['dataset_name']+".xlsx")





