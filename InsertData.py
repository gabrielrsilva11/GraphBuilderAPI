import os
import warnings
from SPARQLWrapper import SPARQLWrapper, POST
from rdflib import Literal
from rdflib.namespace import RDF, OWL
from spacy_conll import init_parser
from tqdm import tqdm
from unidecode import unidecode
from Query_Builder import QueryBuilder


class CreateGraph:
    """
    Main class used to create and manipulate the knowledge graphs

    """

    def __init__(self, doc, graph_name="wikiner_subset_v3", relations_uri={'http://ieeta.pt/ontoud#': 'ieeta'},
                 main_uri='http://ieeta.pt/ontoud#', connection_string='http://localhost:8890/sparql', language="en"):
        """

        :param doc:
        :param graph_name:
        :param relations_uri:
        :param main_uri:
        :param connection_string:
        :param language:
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        self.file_name = doc
        self.file = open(doc, 'r')
        self.graph_name = graph_name
        self.relations_uri_dict = relations_uri
        self.main_uri = main_uri
        self.nlp = init_parser(language,
                               "stanza",
                               ext_names={"conll_pd": "pandas"},
                               parser_opts={"use_gpu": True, "verbose": False},
                               include_headers=True)
        self.relations_uri = relations_uri
        self.connection = connection_string
        self.sparql = SPARQLWrapper(self.connection)
        self.sparql.setMethod(POST)
        self.queries = QueryBuilder(self.relations_uri_dict, self.graph_name)
        # ONTOLOGY URIS
        self.o_contains_sentence = self.main_uri + "containsSentence"
        self.o_from_sentence_uri = self.main_uri + "fromSentence"

        # Text, Sentence, Word classes
        self.c_text_uri = self.main_uri + "Text"
        self.c_sentence_uri = self.main_uri + "Sentence"
        self.c_word_uri = self.main_uri + "Word"

        # HEAD, nextSentence, nextWord, previousWord object properties
        self.o_head_uri = self.main_uri + "head"
        self.o_depgraph_uri = self.main_uri + "depGraph"
        self.o_nextsentence_uri = self.main_uri + "nextSentence"
        self.o_nextword_uri = self.main_uri + "nextWord"
        self.o_previousword_uri = self.main_uri + "previousWord"
        self.o_mapper_uri = self.main_uri + "wikidataId"

        # EDGE, FEATS, ID, LEMMA, POS, POS_COARSE, WORD como data property
        self.d_sentence_text = self.main_uri + "senttext"
        self.d_edge_uri = self.main_uri + "edge"
        self.d_feats_uri = self.main_uri + "feats"
        self.d_id_uri = self.main_uri + "id"
        self.d_lemma_uri = self.main_uri + "lemma"
        self.d_pos_uri = self.main_uri + "pos"
        self.d_poscoarse_uri = self.main_uri + "poscoarse"
        self.d_word_uri = self.main_uri + "word"

    def insert_relationship_data(self):
        """

        :return:
        """
        # class
        self.insert_data(self.c_text_uri, RDF.type, OWL.Class, self.sparql)
        self.insert_data(self.c_sentence_uri, RDF.type, OWL.Class, self.sparql)
        self.insert_data(self.c_word_uri, RDF.type, OWL.Class, self.sparql)

        # object properties
        self.insert_data(self.o_head_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_depgraph_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_nextsentence_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_nextword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_contains_sentence, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_previousword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_from_sentence_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_mapper_uri, RDF.type, OWL.ObjectProperty, self.sparql)

        # data properties
        self.insert_data(self.d_sentence_text, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_edge_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_feats_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_id_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_lemma_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_pos_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_poscoarse_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_word_uri, RDF.type, OWL.DatatypeProperty, self.sparql)

    def insert_data(self, s, p, o, wrapper):
        """

        :param s:
        :param p:
        :param o:
        :param wrapper:
        :return:
        """
        query = self.queries.build_insert_query(s, p, o)
        wrapper.setQuery(query)
        # wrapper.method = 'POST'
        results = wrapper.query()

    def insert_script(self, lines, sentence_id):
        """

        :param lines:
        :param sentence_id:
        :return:
        """
        doc = self.nlp(lines)
        conll = doc._.pandas
        sentence = []
        for index, row in conll.iterrows():
            word = row['form'].replace("'", "").replace("\"", "")
            lemma = row['lemma'].replace("'", "").replace("\"", "")
            word_id = row['id']
            sentence.append(unidecode(word))
            if row['id'] == 1:
                sentenceid_uri = self.c_sentence_uri + "_" + str(sentence_id)
                if sentence_id > 0:
                    new_sentence = sentence[:-1]
                    self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(new_sentence)), self.sparql)
                    sentence = [sentence[-1]]
                    self.insert_data(self.c_text_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                sentence_id += 1
                sentenceid_uri = self.c_sentence_uri + "_" + str(sentence_id)
                wordid_uri = self.d_word_uri + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(sentenceid_uri, RDF.type, self.c_sentence_uri, self.sparql)
                self.insert_data(self.c_text_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                if sentence_id != 1:
                    self.insert_data(self.c_sentence_uri + "_" + str(sentence_id - 1), self.o_nextsentence_uri,
                                     sentenceid_uri, self.sparql)
            else:
                word_id = row['id']
                previous_uri = wordid_uri
                wordid_uri = self.d_word_uri + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(wordid_uri, self.o_previousword_uri, previous_uri, self.sparql)

            self.insert_data(wordid_uri, RDF.type, self.c_word_uri, self.sparql)
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['id']), self.sparql)
            self.insert_data(wordid_uri, self.d_word_uri, Literal(word), self.sparql)
            self.insert_data(wordid_uri, self.d_edge_uri, Literal(row['deprel']), self.sparql)
            self.insert_data(wordid_uri, self.d_feats_uri, Literal(row['feats']), self.sparql)
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['id']), self.sparql)
            self.insert_data(wordid_uri, self.d_lemma_uri, Literal(lemma), self.sparql)
            self.insert_data(wordid_uri, self.d_pos_uri, Literal(row['upostag']), self.sparql)
            self.insert_data(wordid_uri, self.d_poscoarse_uri, Literal(row['xpostag']), self.sparql)

            if row['head'] == 0:
                # print(sentence)
                self.insert_data(wordid_uri, self.o_from_sentence_uri, sentenceid_uri, self.sparql)
                self.insert_data(sentenceid_uri, self.o_depgraph_uri, wordid_uri, self.sparql)
            else:
                self.insert_data(wordid_uri, self.o_head_uri,
                                 self.d_word_uri + "_" + str(sentence_id) + "_" + str(row['head']), self.sparql)
                self.insert_data(self.d_word_uri + "_" + str(sentence_id) + "_" + str(row['head']), self.o_depgraph_uri,
                                 wordid_uri, self.sparql)
        self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(sentence)), self.sparql)
        self.insert_data(self.c_text_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
        return sentence_id

    def insert_wikimapper_data(self, sentence_id, wiki_id):
        """

        :param sentence_id:
        :param wiki_id:
        :return:
        """
        sentence_uri = self.c_sentence_uri + "_" + str(sentence_id)
        query = self.queries.build_insert_wikimapper_query(sentence_uri, self.o_mapper_uri, Literal(wiki_id))
        self.sparql.setQuery(query)
        results = self.sparql.query()

    def create_graph(self):
        """

        :return:
        """
        sentence_id = 0
        i = 0
        lines = ''
        with tqdm(total=os.path.getsize(self.file_name)) as pbar:
            with open(self.file_name) as file:
                for line in file:
                    lines = lines + line
                    if i == 10:
                        pbar.update(len(lines.encode('utf-8')))
                        # pbar.display()
                        i = 0
                        sentence_id = self.insert_script(lines, sentence_id)
                        sentence_id = sentence_id + 1
                        lines = ''
                    i += 1
                if lines:
                    sentence_id = self.insert_script(lines, sentence_id)
        #     for line in self.file:
        #         lines = lines + line
        #         if i == 1000:
        #             i = 0
        #             sentence_id = self.insert_script(lines, sentence_id)
        #             lines = ''
        #         i += 1
        #
        # sentence_id = self.insert_script(lines, sentence_id)

# relations_uri = {"http://ieeta.pt/ontoud#" : "ieeta"}
# connection = 'http://localhost:8890/sparql'
# graph = CreateGraph(doc="data/wikiner_subset.txt", relations_uri=relations_uri,
#                   connection_string=connection, language='pt')
# graph.create_graph()
