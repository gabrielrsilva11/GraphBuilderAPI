from PromptCaRBB import FILE_CONVERSION_PROMPT
import ollama

def query(content):
    # Posso mudar aqui os modelos para o que quiser que ele funciona
    response = ollama.generate(model='qwen2.5:14b-instruct-q5_K_M', prompt=content)
    return response['response']


file = open("/home/grsilva/GraphBuilderAPI_v2/Data_to_process/OpenIE/CaRB/gold_dev.csv", "r")
first_line = True
for line in file:
    if first_line:
        first_line = False
    else:
        split_line = line.split(',')
        a1 = split_line[-5]
        r = split_line[-6]
        a2 = split_line[-7]
        sentence = ''.join(word for word in split_line[:-4])
        print(sentence)
        prompt = FILE_CONVERSION_PROMPT.format(sentence=sentence, a1=a1, r=r, a2=a2)
        print(prompt)
        print(query(prompt))
        break

