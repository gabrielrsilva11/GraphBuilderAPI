import spacy
import re
import spacy_conll
# text = """
#     Do not forget about Momofuku Ando!
#     He created instant noodles in Osaka.
#     At that location, Nissin was founded.
#     Many students survived by eating these noodles, but they don't even know him."""


text = """Here we elaborate on the implementation details of FIMTrack and give an in - depth explanation of the used algorithms .
"""

nlp = spacy.load("en_core_web_sm")
config = {"ext_names": {"conll_pd": "pandas"}}
# nlp.add_pipe(
#     "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": 0})
nlp.add_pipe("conll_formatter", config=config, last=True)
text_nohtml = re.sub(r'http\S+', '', text)
text_nohtml = text_nohtml.lower()
doc = nlp(text_nohtml)
for index, row in doc._.pandas.iterrows():
    if row['LEMMA'] not in nlp.Defaults.stop_words:
        print(row)
