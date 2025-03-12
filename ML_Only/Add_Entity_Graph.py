from openai import AsyncOpenAI
from Prompts.extract_graph import ENTITY_EXTRACTION_PROMPT
from GraphFetch import FetchGraph
from rdflib import URIRef, Literal
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, POST
from Query_Builder import QueryBuilder
import asyncio
import random
from API_Keys import ds_key

async def get_deepseek_response(client, sentence):
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert system in identifying entities in Sentences."},
            {"role": "user", "content": sentence},
        ],
        stream=False
    )
    return response.choices[0].message.content


async def query_deepseek(client, queries):
    tasks = [get_deepseek_response(client, query_sentence) for query_sentence in queries]
    results = await asyncio.gather(*tasks)
    return results



def filter_response(response):
    responses_split = response.split("<nextEntity>")
    entities_dict = {}
    for resp in responses_split:
        resp_split = resp.split("<tuple>")
        if len(resp_split) > 1:
            entities_dict[resp_split[1].lower()] = resp_split[2].lower()
    return entities_dict


def insert_data(s, p, o):
    """
    Inserts a triple into a triple-storage and a knowledge graph.

    :param s: subject of the triple
    :param p: predicate of the triple
    :param o: object of the triple
    :return:
    """
    query = queries.build_insert_query(s, p, o)
    #print(query)
    sparql.setQuery(query)
    # wrapper.method = 'POST'
    for i in range(0, 10):
        try:
            results = sparql.query()
            str_error = None
        except:
            str_error = "Error"
            pass

        if str_error:
            if i == 0:
                print(query)
            print("Error occurred. Attempting to upload triple again.")
            print("Attempt number: ", i)
        else:
            break

client = AsyncOpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
base_uri = "http://www.ieeta-bit.pt/SOMD2025_Tests#"
graph_name = "http://www.ieeta-bit.pt/SOMD2025_Tests#"
conection_string = "http://hlt.ieeta.pt:8890/sparql"
sparql = SPARQLWrapper(conection_string)
sparql.setMethod(POST)
ntx = FetchGraph(base_uri, graph_name, conection_string)
queries = QueryBuilder(base_uri, graph_name)

# ids_to_fetch = range(1, 1150)
start = 1
for j in tqdm(range(11, 204, 10)):
    ids_to_fetch = range(start, j)
    print(start, j)
    queries_list = []
    for i in tqdm(range(len(ids_to_fetch)), desc="Loading Dataset"):
        idx = ids_to_fetch[i]
        graph = ntx.fetch_graph([idx])

        for s, p, o in graph.triples((None, URIRef(base_uri+"senttext"), None)):
            #print(s, p, o)
            sentence = o.__str__()
            query = ENTITY_EXTRACTION_PROMPT.format(input_text=sentence,
                                                    tuple_delimiter="<tuple>",
                                                    completion_delimiter="<end>",
                                                    record_delimiter="<nextEntity>", certainty_delimiter="<rating>")
            queries_list.append(query)
            # response = client.chat.completions.create(
            #     model="deepseek-chat",
            #     messages=[
            #         {"role": "system", "content": "You are an expert system in identifying entities in Sentences."},
            #         {"role": "user", "content": query},
            #     ],
            #     stream=False
            # )
            #
            # response = response.choices[0].message.content
    #TODO: CORRIGIR ISTO E METER A ADICIONAR TODOS EM CONDIÇÕES
    results = asyncio.run(query_deepseek(client, queries_list))
    for response in results:
        entities = filter_response(response)
        ids_to_add = []
        type_to_add = []
        for s, p, o in graph.triples((None, URIRef(base_uri+"word"), None)):
            for key, value in entities.items():
                key_list = key.split(" ")
                if o.__str__().lower() in key_list:
                    #print(value)
                    ids_to_add.append(s)
                    type_to_add.append(value)

        for k in range(0, len(ids_to_add)):
            insert_data(ids_to_add[k], base_uri+"entityType", Literal(type_to_add[k]))
    start = j

