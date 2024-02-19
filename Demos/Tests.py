# from crosslingual_coreference import Predictor
#
# text = ("""
# Here we report a comprehensive suite for the well - known Poisson - Boltzmann solver , DelPhi , enriched with additional features to facilitate DelPhi usage .
# The resource is available free of charge for academic users from URL : http://compbio.clemson.edu/DelPhi.php .
# In this work , we described the DelPhi package and associated resources .
# " Project name : DelPhi Project home page : e.g. http://compbio.clemson.edu/delphi.php Operating system ( s ) : Linux , Mac , Windows Programming language : Fortran and C Other requirements : no License : free of charge license is required Any restrictions to use by non - academics : Commercial users should contact Accelrys Inc . "
# """
# )
#
# # choose minilm for speed/memory and info_xlm for accuracy
# predictor = Predictor(
#     language="en_core_web_sm", device=-1, model_name="minilm"
# )
#
# print(predictor.predict(text)["resolved_text"])
# print(predictor.pipe([text])[0]["resolved_text"])

import spacy

text = """
    Do not forget about Momofuku Ando!
    He created instant noodles in Osaka.
    At that location, Nissin was founded.
    Many students survived by eating these noodles, but they don't even know him."""

# use any model that has internal spacy embeddings
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": 0})

doc = nlp(text)

print(doc._.coref_clusters)
