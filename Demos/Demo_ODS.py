import requests
from pprint import pprint

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
headers = {"Authorization": ""}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

sentences_example = ["De igual modo, este curso visa a formação de técnicos com competências para uma inserção adequada nas estruturas regionais e autárquicas, assim como em organizações que tenham com a Administração Pública, aos seus vários níveis, relações de grande interdependência.",
                     "Contribuir para o desenvolvimento socioeconómico, com especial foco no fortalecimento das Pequenas e Médias Empresas, através da formação de profissionais capazes de contribuir para o crescimento e eficiência das organizações e da economia em geral."]

ods = ["Erradicar a pobreza em todas as suas formas. Erradicar a pobreza extrema para todas as pessoas em todos os lugares. Reduzir pelo menos pela metade a proporção de homens, mulheres e crianças de todas as idades que vivem na pobreza. Implementar sistemas e medidas de proteção social nacionalmente apropriados para todos. Garantir que todos têm direitos iguais aos recursos económicos e acesso a serviços básicos. ",
       "Erradicar a fome em todo o mundo. Garantir o acesso de todas as pessoas a alimentos seguros, nutritivos e suficientes durante todo o ano. Acabar com todas as formas de desnutrição. Garantir sistemas sustentáveis ​​de produção de alimentos. Implementar práticas agrícolas resilientes que aumentem a produtividade e a produção e que ajudem a manter os ecossistemas. ",
       "Garantir saúde e bem-estar para todos. Reduzir a taxa de mortalidade global. Acabar com as mortes evitáveis de recém-nascidos e crianças menores de 5 anos. Erradicar as epidemias de HIV, tuberculose, malária e doenças tropicais negligenciadas. Promover a saúde mental e o bem-estar. Alcançar a cobertura universal de saúde. ",
       "Garantir uma educação inclusiva e de qualidade para todos. Promover a aprendizagem ao longo da vida. Eliminar as disparidades de género na educação. Garantir que todas as meninas e meninos tenham acesso a cuidados e desenvolvimento de qualidade na primeira infância. Garantir a igualdade de acesso a todos os níveis de educação para os mais vulneráveis, incluindo pessoas com deficiência, povos indígenas e crianças em situação de vulnerabilidade. ",
       "Acabar com todas as formas de discriminação contra todas as mulheres e meninas em todos os lugares. Eliminar todas as formas de violência contra todas as mulheres e meninas nas esferas pública e privada, incluindo tráfico, exploração sexual e outros tipos. Garantir a participação plena e efetiva das mulheres e a igualdade de oportunidades de liderança. ",
       "Alcançar o acesso universal à água potável segura e acessível para todos. Alcançar o acesso a saneamento e higiene adequados para todos. Melhorar a qualidade da água reduzindo a poluição, eliminando o despejo de produtos químicos e materiais perigosos. ",
       "Garantir o acesso universal a energia renovável e acessível para todos. Aumentar a participação das energias renováveis no mix global de energia. Reforçar a cooperação internacional para facilitar o acesso à pesquisa e tecnologia de energia limpa. Expandir a infraestrutura e atualizar a tecnologia para fornecer serviços de energia modernos e sustentáveis para todos nos países em desenvolvimento. ",
       "Garantir o desenvolvimento económico inclusivo e sustentável em todo o mundo. Alcançar níveis mais altos de produtividade económica por meio da diversificação, atualização tecnológica e inovação. Alcançar emprego pleno e produtivo e trabalho decente para todos, inclusive para jovens e pessoas com deficiência. Alcançar salário igual para trabalho de igual valor. ",
       "Garantir a inovação e infraestruturas sustentáveis da indústria. Desenvolver infraestrutura confiável, sustentável e resiliente que apoie o desenvolvimento económico e o bem-estar humano. Promover a industrialização inclusiva e sustentável. Atualizar as infraestruturas e modernizar as indústrias para torná-las sustentáveis. Apoiar uma maior adoção de tecnologias renováveis. ",
       "Capacitar e promover a inclusão social, económica e política de todos, independentemente de idade, sexo, deficiência, raça, etnia, origem, religião ou condição econômica ou outra. Garanta a igualdade de oportunidades. Reduzir as desigualdades de resultado, eliminando leis, políticas e práticas discriminatórias. ",
       "Construir cidades e sociedades sustentáveis em todo o mundo. Garantir o acesso de todos a uma habitação adequada, segura e acessível. Aumentar a capacidade de planeamento e gestão integrados e sustentáveis de aglomerados humanos. Reduzir o impacto ambiental adverso das cidades, prestando atenção especial à qualidade do ar e à gestão de resíduos. ",
       "Reduzir o desperdício global de alimentos na produção e consumidor. Alcançar a gestão ambientalmente saudável de produtos químicos ao longo de seu ciclo de vida. Reduzir substancialmente a geração de resíduos por meio da prevenção, redução, reciclagem e reutilização.",
       "Tomar medidas urgentes para combater as mudanças climáticas e seus impactos. Fortalecer a resiliência e a capacidade de adaptação aos perigos e desastres naturais relacionados ao clima. Integrar soluções e medidas de mudança climática nas políticas, estratégias e planejamento nacionais. Melhorar a educação sobre mitigação das mudanças climáticas, redução de impacto e alerta precoce. ",
       "Conservar e usar de forma sustentável os oceanos, mares e recursos marinhos. Prevenir e diminuir a poluição marinha de todos os tipos, em particular de atividades terrestres. Gerir e proteger de forma sustentável os ecossistemas marinhos e costeiros para evitar impactos adversos significativos. Acabar com a sobrepesca, práticas de pesca ilegais, não declaradas e destrutivas. ",
       "Prevenir ameaças à biodiversidade.Garantir a conservação, restauração e uso sustentável dos ecossistemas terrestres e de água doce, incluindo florestas, pântanos, montanhas e terras secas.Promover a implementação da gestão sustentável de todos os tipos de florestas.Deter o desmatamento.Combater a desertificação e restaurar terras e solos degradados. ",
       "Promover sociedades justas, pacíficas e inclusivas.Reduzir significativamente todas as formas de violência.Erradicar o abuso, a exploração e o tráfico e todas as formas de violência e tortura de crianças.Promover o estado de direito nos níveis nacional e internacional.Garantir a igualdade de acesso à justiça para todos. " ,
       "Revitalizar a parceria global para o desenvolvimento sustentável.Apoiar a criação de fortes parcerias ODS para atingir as metas ambiciosas da Agenda 2030.Reúna os governos nacionais, a comunidade internacional, a sociedade civil, o setor privado e outros atores. "]

ods_dict = {1: 'Erradicar a Pobreza',
            2: 'Erradicar a Fome',
            3: 'Saúde de Qualidade',
            4: 'Educação de Qualidade',
            5: 'Igualdade de Género',
            6: 'Água Potável e Saneamento',
            7: 'Energias Renováveis e Acessíveis',
            8: 'Trabalho Digno e Crescimento Económico',
            9: 'Indústria, Inovação e Infraestruturas',
            10: 'Reduzir as Desigualdades',
            11: 'Cidades e Comunidades Sustentáveis',
            12: 'Produção e Consumo Sustentáveis',
            13: 'Ação Climaática',
            14: 'Proteger a Vida Marinha',
            15: 'Proteger a Vida Terrestre',
            16: 'Paz, Justiça e Instituições Eficazes',
            17: 'Parcerias para a Implementação dos Objetivos'}

for sentence in sentences_example:
    output = query({
        "inputs": {
            "source_sentence": sentence,
            "sentences": ods
        },
    })
    list_to_sort = []
    for i in range(0, len(output)):
        #print(ods_dict[i+1], ": ", output[i])
        list_to_sort.append(str(output[i]) + ' ' + str(ods_dict[i+1]))

    # print(list_to_sort)
    # sorted_list = list_to_sort.sort()
    index_max = max(range(len(output)), key=output.__getitem__)
    list_to_sort.sort(reverse=True)
    pprint(list_to_sort)
    print("MAIS PROVAVÉL: ", ods_dict[index_max + 1], output[index_max])
    print("------------------------------")
