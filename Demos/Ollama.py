import ollama
import json


def query(content):
    # Posso mudar aqui os modelos para o que quiser que ele funciona
    response = ollama.generate(model='llama3.2', prompt=content)
    return response['response']


def chat(message_with_context):
    stream = ollama.chat(
        #Posso mudar aqui os modelos para o que quiser que ele funciona
        model='llama3.2',
        messages=message_with_context,
    )
    return stream


with open('QueryHistory/history.json') as f:
    history = json.load(f)

chat_message = chat(history['history'])
response = chat_message['message']['content']
print(response)