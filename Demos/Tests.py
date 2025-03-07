# from spacy_conll import init_parser
#
# nlp = init_parser("en_core_web_sm",
#                                "spacy",
#                                ext_names={"conll_pd": "pandas"},
#                                is_tokenized=True,
#                                #disable_sbd=True,
#                                #parser_opts={"use_gpu": True, "verbose": False},
#                                include_headers=True)
#
# sentence = "The groups were allowed to analyze the training and test data in any way they deemed fit , but all used a combination of the following methods : ( i ) Visual inspection with dynamic varying of thresholds using a software such as Mricron or FSLView . ( ii ) Voxel - wise correlation of brain maps from the training and the test set , to find the blocks which are most similar to each other . ( iii ) Voxel - wise correlations of brain maps with maps from NeuroSynth [33] , to find the keywords from the NeuroSynth database whose posterior probability maps are most similar to the participant ’ s activity patterns ."
#
# doc = nlp(sentence)
# conll = doc._.pandas
# print(conll)

#
# from openai import AsyncOpenAI
# import asyncio
#
#
# async def get_deepseek_response(client, sentence):
#     response = await client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are an expert system in identifying entities in Sentences."},
#             {"role": "user", "content": sentence},
#         ],
#         stream=False
#     )
#     return response.choices[0].message.content
#
#
# async def query_deepseek(client, queries):
#     tasks = [get_deepseek_response(client, query) for query in queries]
#     results = await asyncio.gather(*tasks)
#     return results
#
# queries = ["Tests for linkage disequilibrium were checked by using the GenePop v. 4.0 [62] .",
#             "We used IBM SPSS 22.0 for the quantitative analysis ."]
# client = AsyncOpenAI(api_key="sk-547e4d33b0684d8d84443f7cf63afe61", base_url="https://api.deepseek.com")
# results = asyncio.run(query_deepseek(client, queries))
# print(results)
# query = ENTITY_EXTRACTION_PROMPT.format(input_text=sentence,
#                                         tuple_delimiter="<tuple>",
#                                         completion_delimiter="<end>",
#                                         record_delimiter="<nextEntity>", certainty_delimiter="<rating>")

import numpy as np

from sklearn.utils.class_weight import compute_class_weight

y = {'No':               18203,
'Application':                1198,
'Developer':                   947,
'Version':                     648,
'Citation':                    309,
'PlugIn':                      221,
'ProgrammingEnvironment':      175,
'URL':                         150,
'OperatingSystem':             110,
'License':                      71,
'Release':                      53,
'AlternativeName':              43,
'Extension':                    37,
'Abbreviation':                 36,
'SoftwareCoreference':          18}

A = [18203, 1198, 947, 150, 648, 221, 309, 37, 175, 110, 53, 36, 71, 18, 43]
# total_samples / (num_samples_in_class_i * num_classes)
soma = sum(A)
class_weights = [soma/num_samples*15 for num_samples in A]
print(class_weights)
