import spacy
import re
from fastcoref import spacy_component
#from GraphCreation_v2 import preprocess_sentence

# text = """
#     Do not forget about Momofuku Ando!
#     He created instant noodles in Osaka.
#     At that location, Nissin was founded.
#     Many students survived by eating these noodles, but they don't even know him."""


text = """Here we report a comprehensive suite for the well - known Poisson - Boltzmann solver , DelPhi , enriched with additional features to facilitate DelPhi usage .
The resource is available free of charge for academic users from URL : http://compbio.clemson.edu/DelPhi.php .
In this work , we described the DelPhi package and associated resources .
"""

nlp = spacy.load("en_core_web_trf")

config = {"ext_names": {"conll_pd": "pandas"}}
nlp.add_pipe("conll_formatter", config=config, last=True)
nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref'})

text_nohtml = re.sub(r'http\S+', '', text)
# text_nohtml = text_nohtml.lower()
doc = nlp(text_nohtml, component_cfg={"fastcoref": {'resolve_text': True}})
for index, row in doc._.pandas.iterrows():
    if row['LEMMA'] not in nlp.Defaults.stop_words:
        print(row)

print(doc._.coref_clusters)
print(doc._.resolved_text)

