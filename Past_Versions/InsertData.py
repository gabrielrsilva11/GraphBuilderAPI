import os
from time import sleep
import warnings
from SPARQLWrapper import SPARQLWrapper, POST
from spacy_conll import init_parser
from tqdm import tqdm
from unidecode import unidecode
from Query_Builder import QueryBuilder
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF, DOAP, FOAF, ORG, OWL, SKOS, XSD
import string


class CreateGraph:
    """
    Main class used to create the knowledge graphs.

    """

    def __init__(self, folder, graph_name, extra_connetions = [], main_uri='http://ieeta.pt/ontoud#',
                    connection_string='http://localhost:8890/sparql', language="pt_core_news_sm", preprocessing = None, in_memory = False):
        """
        Instantiates a CreateGraph class.
        :param doc: path to the document or folder of documents to process.
        :param graph_name: name to give the knowledge graph.
        :param relations_uri:
        :param main_uri:
        :param connection_string: the connection string that is used to connect to a triple storage
        :param language: language in which the text is in.
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        self.folder_name = folder
        self.graph_name = graph_name
        self.main_uri = main_uri
        self.in_memory = in_memory
        self.preprocessing = preprocessing
        self.nlp = init_parser(language,
                               "spacy",
                               ext_names={"conll_pd": "pandas"},
                               disable_sbd=True,
                               #parser_opts={"use_gpu": True, "verbose": False},
                               include_headers=True)
        self.connection = connection_string
        self.sparql = SPARQLWrapper(self.connection)
        self.sparql.setMethod(POST)
        self.queries = QueryBuilder(self.main_uri, self.graph_name)

        # Document navigation -> Text, Sentence, Word classes
        self.c_text_uri = self.main_uri + "Text"
        self.c_sentence_uri = self.main_uri + "Sentence"
        self.c_word_uri = self.main_uri + "Word"

        # General properties -> text/conll properties to be created as object properties in the graph
        self.o_depgraph_uri = self.main_uri + "depGraph"
        self.o_nextsentence_uri = self.main_uri + "nextSentence"
        self.o_previoussentence_uri = self.main_uri + "previousSentence"
        self.o_nextword_uri = self.main_uri + "nextWord"
        self.o_previousword_uri = self.main_uri + "previousWord"
        self.o_mapper_uri = self.main_uri + "wikidataId"
        self.o_contains_sentence = self.main_uri + "containsSentence"
        self.o_from_text = self.main_uri + "fromText"
        self.o_contains_text = self.main_uri + "containsText"
        self.o_from_sentence_uri = self.main_uri + "fromSentence"

        #Extra object properties
        self.extra_object_properties = self.fetch_extra_properties(extra_connetions)

        # CoNLL properties -> EDGE, FEATS, ID, LEMMA, POS, POS_COARSE, WORD as a data property
        self.o_head_uri = self.main_uri + "head"
        self.d_sentence_text = self.main_uri + "senttext"
        self.d_edge_uri = self.main_uri + "edge"
        #self.d_feats_uri = self.main_uri + "feats"
        self.d_id_uri = self.main_uri + "id"
        self.d_lemma_uri = self.main_uri + "lemma"
        self.d_pos_uri = self.main_uri + "pos"
        self.d_poscoarse_uri = self.main_uri + "poscoarse"
        self.d_word_uri = self.main_uri + "word"
        #dict to keep track of the already inserted feats
        self.d_feats_list = []

    def fetch_extra_properties(self, extra_connetions):
        connections_list = []
        for name in extra_connetions:
            connections_list.append(self.main_uri+name)
        print(connections_list)
        return connections_list

    def insert_db_relationship_data(self):
        """
        Creates and adds to the graph the CoNLL relationship properties.
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
        self.insert_data(self.o_previoussentence_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_nextword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_previousword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_contains_text, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_contains_sentence, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_from_text, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_previousword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_from_sentence_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_mapper_uri, RDF.type, OWL.ObjectProperty, self.sparql)

        #Insert the extras
        if self.extra_object_properties:
            for extra_object in self.extra_object_properties:
                self.insert_data(extra_object, RDF.type, OWL.ObjectProperty, self.sparql)

        # data properties
        self.insert_data(self.d_sentence_text, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_edge_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        #self.insert_data(self.d_feats_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_id_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_lemma_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_pos_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_poscoarse_uri, RDF.type, OWL.DatatypeProperty, self.sparql)
        self.insert_data(self.d_word_uri, RDF.type, OWL.DatatypeProperty, self.sparql)

    def insert_memory_relationship_data(self, g):
        """
        Creates and adds to the graph the CoNLL relationship properties.
        :return:
        """
        g.bind("rdfs", RDFS)
        g.bind("rdf", RDF)
        g.bind("doap", DOAP)
        g.bind("org", ORG)
        g.bind("owl", OWL)
        g.bind("skos", SKOS)
        g.bind("xsd", XSD)
        g.bind("foaf", FOAF)

        # class
        g.add((URIRef(self.c_text_uri), RDF.type, OWL.Class))
        g.add((URIRef(self.c_sentence_uri), RDF.type, OWL.Class))
        g.add((URIRef(self.c_word_uri), RDF.type, OWL.Class))

        # object properties
        g.add((URIRef(self.o_head_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_depgraph_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_nextsentence_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_nextword_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_previoussentence_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_previousword_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_contains_text), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_contains_sentence), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_from_text), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_previousword_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_from_sentence_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_mapper_uri), RDF.type, OWL.ObjectProperty))

        #Insert the extras
        if self.extra_object_properties:
            for extra_object in self.extra_object_properties:
                g.add((URIRef(extra_object), RDF.type, OWL.DatatypeProperty))

        # data properties
        g.add((URIRef(self.d_sentence_text), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_edge_uri), RDF.type, OWL.DatatypeProperty))
        #g.add((URIRef(self.d_feats_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_id_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_lemma_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_pos_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_poscoarse_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_word_uri), RDF.type, OWL.DatatypeProperty))

    def insert_data(self, s, p, o, wrapper):
        """
        Inserts a triple into a triple-storage and a knowledge graph.

        :param s: subject of the triple
        :param p: predicate of the triple
        :param o: object of the triple
        :param wrapper: connection to the triple storage
        :return:
        """
        query = self.queries.build_insert_query(s, p, o)
        wrapper.setQuery(query)
        # wrapper.method = 'POST'
        for i in range(0, 10):
            try:
                results = wrapper.query()
                str_error = None
            except:
                str_error = "Error"
                pass
            if str_error:
                print("Error occurred. Attempting to upload triple again.")
                print("Attempt number: ", i)

                sleep(2*i)  # wait for 2*attempt number seconds before trying to fetch the data again
            else:
                break

    def insert_db_script(self, lines, sentence_id, doc_id):
        """
        Main script to insert CoNLL data into a triple-storage.
        :param lines: the text to insert
        :param sentence_id: last known sentence_id for identification purposes.
        :param doc_id: the id of the document we are currently processing
        :return: the last used sentence_id.
        """
        if self.preprocessing:
            processed_lines = self.preprocessing(lines)
            sentence = ""
            for line in processed_lines:
                sentence += line[0] + " "
        else:
            sentence = lines

        sentence = sentence.strip()
        doc = self.nlp(sentence)
        conll = doc._.pandas
        sentence = []
        textid_uri = URIRef(self.c_text_uri + "_" + str(doc_id))
        self.insert_data(textid_uri, RDF.type, self.c_text_uri, self.sparql)
        indexes_used = []
        for index, row in conll.iterrows():
            word = row['FORM'].replace("'", "").replace("\"", "")
            lemma = row['LEMMA'].replace("'", "").replace("\"", "")
            word_id = row['ID']
            sentence.append(unidecode(word))
            if row['ID'] == 1:
                sentenceid_uri = self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id)
                if sentence_id > 0:
                    sentence = [sentence[-1]]
                    self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                    self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(sentenceid_uri, RDF.type, self.c_sentence_uri, self.sparql)
                self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
                if sentence_id != 1:
                    #Previous sentence
                    self.insert_data(sentenceid_uri, self.o_nextsentence_uri,
                                     self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1), self.sparql)
                    #Next sentence
                    self.insert_data(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1), self.o_nextsentence_uri,
                                     sentenceid_uri, self.sparql)
            else:
                word_id = row['ID']
                previous_uri = wordid_uri
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(wordid_uri, self.o_previousword_uri, previous_uri, self.sparql)
                self.insert_data(previous_uri, self.o_nextword_uri, wordid_uri, self.sparql)

            self.insert_data(wordid_uri, RDF.type, self.c_word_uri, self.sparql)
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['ID']), self.sparql)
            self.insert_data(wordid_uri, self.d_word_uri, Literal(word), self.sparql)
            self.insert_data(wordid_uri, self.d_edge_uri, Literal(row['DEPREL']), self.sparql)
            self.process_feats(wordid_uri, row['FEATS'])
            #self.insert_data(wordid_uri, self.d_feats_uri, Literal(row['feats']), self.sparql)
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['ID']), self.sparql)
            self.insert_data(wordid_uri, self.d_lemma_uri, Literal(lemma), self.sparql)
            self.insert_data(wordid_uri, self.d_pos_uri, Literal(row['UPOS']), self.sparql)
            transformed = ''
            for character in row['XPOS']:
                if character in string.punctuation:
                    transformed = transformed + str(ord(character))
                else:
                    transformed = transformed + character
            self.insert_data(wordid_uri, self.d_poscoarse_uri, Literal(transformed), self.sparql)
            if self.preprocessing:
                for o in range(0, len(processed_lines)):
                    word_to_check = processed_lines[o][0].strip().lower()
                    if word.lower() in word_to_check and o <= index and o not in indexes_used:
                        indexes_used.append(o)
                        for k in range(0, len(self.extra_object_properties)):
                            if processed_lines[o][k + 1] != '':
                                self.insert_data(wordid_uri, URIRef(self.extra_object_properties[k]),
                                       Literal(processed_lines[o][k + 1]), self.sparql)
                        break
            if row['HEAD'] == 0:
                # print(sentence)
                self.insert_data(wordid_uri, self.o_from_sentence_uri, sentenceid_uri, self.sparql)
                self.insert_data(sentenceid_uri, self.o_depgraph_uri, wordid_uri, self.sparql)
            else:
                self.insert_data(wordid_uri, self.o_head_uri,
                                 self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['HEAD']), self.sparql)
                self.insert_data(self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['HEAD']), self.o_depgraph_uri,
                                 wordid_uri, self.sparql)
        self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(sentence)), self.sparql)
        self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
        self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
        return sentence_id

    def process_feats(self, wordid_uri, feats, g = None):
        split_feats = feats.split("|")
        if split_feats[0] != "_":
            for feat in split_feats:
                feat = feat.split("=")
                if feat[0] in self.d_feats_list:
                    if self.in_memory:
                        g.add((wordid_uri, URIRef(self.main_uri+feat[0].lower()), Literal(feat[1])))
                    else:
                        self.insert_data(wordid_uri, self.main_uri+feat[0].lower(), Literal(feat[1]), self.sparql)
                else:
                    if self.in_memory:
                        self.d_feats_list.append(feat[0])
                        g.add((URIRef(self.main_uri+feat[0].lower()), RDF.type, OWL.DatatypeProperty))
                        g.add((wordid_uri, URIRef(self.main_uri+feat[0].lower()), Literal(feat[1])))
                    else:
                        self.d_feats_list.append(feat[0])
                        self.insert_data(self.main_uri+feat[0].lower(), RDF.type, OWL.DatatypeProperty, self.sparql)
                        self.insert_data(wordid_uri, self.main_uri+feat[0].lower(), Literal(feat[1]), self.sparql)


    def insert_memory_script(self, lines, sentence_id, doc_id, g):
        """
        Main script to insert CoNLL data into a triple-storage.
        :param lines: the text to insert
        :param sentence_id: last known sentence_id for identification purposes.
        :return: the last used sentence_id.
        """
        if self.preprocessing:
            processed_lines = self.preprocessing(lines)
            sentence = ""
            for line in processed_lines:
                sentence += line[0] + " "
        else:
            sentence = lines

        sentence = sentence.strip()
        doc = self.nlp(sentence)
        conll = doc._.pandas
        sentence = []
        textid_uri = URIRef(self.c_text_uri + "_" + str(doc_id))
        g.add((textid_uri, RDF.type, URIRef(self.c_text_uri)))
        indexes_used = []
        for index, row in conll.iterrows():
            word = row['FORM'].replace("'", "").replace("\"", "")
            lemma = row['LEMMA'].replace("'", "").replace("\"", "")
            word_id = row['ID']
            sentence.append(unidecode(word))
            if row['ID'] == 1:
                sentenceid_uri = URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id))
                if sentence_id > 0:
                    sentence = [sentence[-1]]
                    g.add((textid_uri, URIRef(self.o_contains_sentence), sentenceid_uri))
                    g.add((sentenceid_uri, URIRef(self.o_from_text), textid_uri))
                #sentence_id += 1
                wordid_uri = URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id))
                g.add((sentenceid_uri, RDF.type, URIRef(self.c_sentence_uri)))
                g.add((textid_uri, URIRef(self.o_contains_sentence), sentenceid_uri))
                g.add((sentenceid_uri, URIRef(self.o_from_text), textid_uri))
                if sentence_id != 1:
                    #Previous Sentence
                    g.add((URIRef(sentenceid_uri), URIRef(self.o_previoussentence_uri),
                                     URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id-1))))
                    # #Next sentence
                    g.add((URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1)),
                           URIRef(self.o_nextsentence_uri), sentenceid_uri))
            else:
                word_id = row['ID']
                previous_uri = wordid_uri
                wordid_uri = URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id))
                g.add((wordid_uri, URIRef(self.o_previousword_uri), previous_uri))
                g.add((previous_uri, URIRef(self.o_nextword_uri), wordid_uri))

            g.add((wordid_uri, RDF.type, URIRef(self.c_word_uri)))
            g.add((wordid_uri, URIRef(self.d_id_uri), Literal(row['ID'])))
            g.add((wordid_uri, URIRef(self.d_word_uri), Literal(word)))
            g.add((wordid_uri, URIRef(self.d_edge_uri), Literal(row['DEPREL'])))
            self.process_feats(wordid_uri, row['FEATS'], g)
            #g.add((wordid_uri, URIRef(self.d_feats_uri), Literal(row['feats'])))
            g.add((wordid_uri, URIRef(self.d_id_uri), Literal(row['ID'])))
            g.add((wordid_uri, URIRef(self.d_lemma_uri), Literal(lemma)))
            g.add((wordid_uri, URIRef(self.d_pos_uri), Literal(row['UPOS'])))
            #Processing POSCOARSE
            transformed = ''
            for character in row['XPOS']:
                if character in string.punctuation:
                    transformed = transformed + str(ord(character))
                else:
                    transformed = transformed + character
            g.add((wordid_uri, URIRef(self.d_poscoarse_uri), Literal(transformed)))
            if self.preprocessing:
                for o in range(0, len(processed_lines)):
                    word_to_check = processed_lines[o][0].strip().lower()
                    if word.lower() in word_to_check and o <= index and o not in indexes_used:
                        indexes_used.append(o)
                        for k in range(0, len(self.extra_object_properties)):
                            if processed_lines[o][k + 1] != '':
                                g.add((wordid_uri, URIRef(self.extra_object_properties[k]),
                                       Literal(processed_lines[o][k + 1])))
                        break

            if row['HEAD'] == 0:
                g.add((wordid_uri, URIRef(self.o_from_sentence_uri), sentenceid_uri))
                g.add((sentenceid_uri, URIRef(self.o_depgraph_uri), wordid_uri))
            else:
                g.add((wordid_uri, URIRef(self.o_head_uri),
                                 URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(row['HEAD']))))
                g.add((URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(row['HEAD'])), URIRef(self.o_depgraph_uri),
                                 wordid_uri))
        g.add((sentenceid_uri, URIRef(self.d_sentence_text), Literal(' '.join(sentence))))
        g.add((textid_uri, URIRef(self.o_contains_sentence), sentenceid_uri))
        g.add((sentenceid_uri, URIRef(self.o_from_text), textid_uri))
        return sentence_id

    def insert_wikimapper_data(self, sentence_id, wiki_id):
        """
        Builds and inserts a query to insert a triple with wikimapper data into the graph.
        :param sentence_id: id of the sentence that contains the named entity.
        :param wiki_id: wikidata id of the named entity.
        :return:
        """
        sentence_uri = self.c_sentence_uri + "_" + str(sentence_id)
        query = self.queries.build_insert_wikimapper_query(sentence_uri, self.o_mapper_uri, Literal(wiki_id))
        self.sparql.setQuery(query)
        results = self.sparql.query()

    def create_graph(self, save_file = "Serialized"):
        """

        :param in_memory: Boolean which indicates whether we want to create the graph in-memory or upload to a storage.
        :param save_file: Name of the file to save the graph.
        """
        doc_id = 0
        i = 0
        lines = ''
        files = [f for f in os.listdir(self.folder_name) if os.path.isfile(os.path.join(self.folder_name, f))]
        if self.in_memory:
            g = Graph()
            self.insert_memory_relationship_data(g)
        else:
            self.insert_db_relationship_data()

        for file_name in files:
            if not file_name.startswith("."):
                sentence_id = 1
                file_path = os.getcwd()+"/"+self.folder_name+"/"+file_name
                print(f"--- Processing file {doc_id} : {file_name} ---")
                with tqdm(total=os.path.getsize(file_path)) as pbar:
                    with open(file_path) as file:
                        for line in file:
                            #print(repr(line))
                            if line != "\n":
                                lines = lines + line
                                if self.in_memory:
                                    sentence_id = self.insert_memory_script(line, sentence_id, doc_id, g)
                                else:
                                    sentence_id = self.insert_db_script(line, sentence_id, doc_id)
                                sentence_id = sentence_id + 1
                                i += 1
                                if i == 50:
                                    pbar.update(len(lines.encode('utf-8')))
                                    # pbar.display()
                                    i = 0
                                    lines = ''
                        pbar.update(len(lines.encode('utf-8')))
                            # if lines:
                            #     if in_memory:
                            #         sentence_id = self.insert_memory_script(lines, sentence_id, doc_id, g)
                            #     else:
                            #         sentence_id = self.insert_db_script(lines, sentence_id, doc_id)
            doc_id += 1
        if self.in_memory:
            g.serialize(destination=save_file+".ttl", format="turtle")
