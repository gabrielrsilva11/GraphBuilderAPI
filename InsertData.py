import os
import warnings
from SPARQLWrapper import SPARQLWrapper, POST
from spacy_conll import init_parser
from tqdm import tqdm
from unidecode import unidecode
from Query_Builder import QueryBuilder
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF, DOAP, FOAF, ORG, OWL, SKOS, XSD

class CreateGraph:
    """
    Main class used to create and manipulate the knowledge graphs.
    Initially giv

    """

    def __init__(self, folder, graph_name="wikiner_subset_v3", relations_uri={'http://ieeta.pt/ontoud#': 'ieeta'},
                 main_uri='http://ieeta.pt/ontoud#', connection_string='http://localhost:8890/sparql', language="en"):
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

        # Document navigation -> Text, Sentence, Word classes
        self.c_text_uri = self.main_uri + "Text"
        self.c_sentence_uri = self.main_uri + "Sentence"
        self.c_word_uri = self.main_uri + "Word"

        # General properties -> text/conll properties to be created as object properties in the graph
        self.o_depgraph_uri = self.main_uri + "depGraph"
        self.o_nextsentence_uri = self.main_uri + "nextSentence"
        self.o_nextword_uri = self.main_uri + "nextWord"
        self.o_previousword_uri = self.main_uri + "previousWord"
        self.o_mapper_uri = self.main_uri + "wikidataId"
        self.o_contains_sentence = self.main_uri + "containsSentence"
        self.o_from_text = self.main_uri + "fromText"
        self.o_contains_text = self.main_uri + "containsText"
        self.o_from_sentence_uri = self.main_uri + "fromSentence"

        # CoNLL properties -> EDGE, FEATS, ID, LEMMA, POS, POS_COARSE, WORD as a data property
        self.o_head_uri = self.main_uri + "head"
        self.d_sentence_text = self.main_uri + "senttext"
        self.d_edge_uri = self.main_uri + "edge"
        self.d_feats_uri = self.main_uri + "feats"
        self.d_id_uri = self.main_uri + "id"
        self.d_lemma_uri = self.main_uri + "lemma"
        self.d_pos_uri = self.main_uri + "pos"
        self.d_poscoarse_uri = self.main_uri + "poscoarse"
        self.d_word_uri = self.main_uri + "word"

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
        self.insert_data(self.o_nextword_uri, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_contains_text, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_contains_sentence, RDF.type, OWL.ObjectProperty, self.sparql)
        self.insert_data(self.o_from_text, RDF.type, OWL.ObjectProperty, self.sparql)
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
        g.add((URIRef(self.o_contains_text), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_contains_sentence), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_from_text), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_previousword_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_from_sentence_uri), RDF.type, OWL.ObjectProperty))
        g.add((URIRef(self.o_mapper_uri), RDF.type, OWL.ObjectProperty))

        # data properties
        g.add((URIRef(self.d_sentence_text), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_edge_uri), RDF.type, OWL.DatatypeProperty))
        g.add((URIRef(self.d_feats_uri), RDF.type, OWL.DatatypeProperty))
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
        results = wrapper.query()

    def insert_db_script(self, lines, sentence_id, doc_id):
        """
        Main script to insert CoNLL data into a triple-storage.
        :param lines: the text to insert
        :param sentence_id: last known sentence_id for identification purposes.
        :return: the last used sentence_id.
        """
        doc = self.nlp(lines)
        conll = doc._.pandas
        sentence = []
        textid_uri = self.c_text_uri + "_" + str(doc_id)
        self.insert_data(textid_uri, RDF.type, self.c_text_uri, self.sparql)
        for index, row in conll.iterrows():
            word = row['form'].replace("'", "").replace("\"", "")
            lemma = row['lemma'].replace("'", "").replace("\"", "")
            word_id = row['id']
            sentence.append(unidecode(word))
            if row['id'] == 1:
                sentenceid_uri = self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id)
                if sentence_id > 0:
                    new_sentence = sentence[:-1]
                    self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(new_sentence)), self.sparql)
                    sentence = [sentence[-1]]
                    self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                    self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
                sentence_id += 1
                sentenceid_uri = self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id)
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(sentenceid_uri, RDF.type, self.c_sentence_uri, self.sparql)
                self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
                self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
                if sentence_id != 1:
                    self.insert_data(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1), self.o_nextsentence_uri,
                                     sentenceid_uri, self.sparql)
            else:
                word_id = row['id']
                previous_uri = wordid_uri
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
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
                                 self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['head']), self.sparql)
                self.insert_data(self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['head']), self.o_depgraph_uri,
                                 wordid_uri, self.sparql)
        self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(sentence)), self.sparql)
        self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri, self.sparql)
        self.insert_data(sentenceid_uri, self.o_from_text, textid_uri, self.sparql)
        return sentence_id

    def insert_memory_script(self, lines, sentence_id, doc_id, g):
        """
        Main script to insert CoNLL data into a triple-storage.
        :param lines: the text to insert
        :param sentence_id: last known sentence_id for identification purposes.
        :return: the last used sentence_id.
        """
        doc = self.nlp(lines)
        conll = doc._.pandas
        sentence = []
        textid_uri = URIRef(self.c_text_uri + "_" + str(doc_id))
        g.add((textid_uri, RDF.type, URIRef(self.c_text_uri)))
        for index, row in conll.iterrows():
            word = row['form'].replace("'", "").replace("\"", "")
            lemma = row['lemma'].replace("'", "").replace("\"", "")
            word_id = row['id']
            sentence.append(unidecode(word))
            if row['id'] == 1:
                sentenceid_uri = URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id))
                if sentence_id > 0:
                    new_sentence = sentence[:-1]
                    g.add((sentenceid_uri, URIRef(self.d_sentence_text), Literal(' '.join(new_sentence))))
                    sentence = [sentence[-1]]
                    g.add((textid_uri, URIRef(self.o_contains_sentence), sentenceid_uri))
                    g.add((sentenceid_uri, URIRef(self.o_from_text), textid_uri))
                sentence_id += 1
                sentenceid_uri = URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id))
                wordid_uri = URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id))
                g.add((sentenceid_uri, RDF.type, URIRef(self.c_sentence_uri)))
                g.add((textid_uri, URIRef(self.o_contains_sentence), sentenceid_uri))
                g.add((sentenceid_uri, URIRef(self.o_from_text), textid_uri))
                if sentence_id != 1:
                    g.add((URIRef(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1)), URIRef(self.o_nextsentence_uri),
                                     sentenceid_uri))
            else:
                word_id = row['id']
                previous_uri = wordid_uri
                wordid_uri = URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id))
                g.add((wordid_uri, URIRef(self.o_previousword_uri), previous_uri))

            g.add((wordid_uri, RDF.type, URIRef(self.c_word_uri)))
            g.add((wordid_uri, URIRef(self.d_id_uri), Literal(row['id'])))
            g.add((wordid_uri, URIRef(self.d_word_uri), Literal(word)))
            g.add((wordid_uri, URIRef(self.d_edge_uri), Literal(row['deprel'])))
            g.add((wordid_uri, URIRef(self.d_feats_uri), Literal(row['feats'])))
            g.add((wordid_uri, URIRef(self.d_id_uri), Literal(row['id'])))
            g.add((wordid_uri, URIRef(self.d_lemma_uri), Literal(lemma)))
            g.add((wordid_uri, URIRef(self.d_pos_uri), Literal(row['upostag'])))
            g.add((wordid_uri, URIRef(self.d_poscoarse_uri), Literal(row['xpostag'])))

            if row['head'] == 0:
                # print(sentence)
                g.add((wordid_uri, URIRef(self.o_from_sentence_uri), sentenceid_uri))
                g.add((sentenceid_uri, URIRef(self.o_depgraph_uri), wordid_uri))
            else:
                g.add((wordid_uri, URIRef(self.o_head_uri),
                                 URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(row['head']))))
                g.add((URIRef(self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(row['head'])), URIRef(self.o_depgraph_uri),
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

    def create_graph(self, in_memory=False, save_file = "Serialized"):
        """

        :param in_memory: Boolean which indicates whether we want to create the graph in-memory or upload to a storage.
        """
        doc_id = 0
        i = 0
        lines = ''
        files = [f for f in os.listdir(self.folder_name) if os.path.isfile(os.path.join(self.folder_name, f))]
        if in_memory:
            g = Graph()
            self.insert_memory_relationship_data(g)
        else:
            self.insert_db_relationship_data()

        for file_name in files:
            if not file_name.startswith("."):
                sentence_id = 0
                file_path = os.getcwd()+"/"+self.folder_name+"/"+file_name
                print(f"--- Processing file {doc_id} : {file_name} ---")
                with tqdm(total=os.path.getsize(file_path)) as pbar:
                    with open(file_path) as file:
                        for line in file:
                            lines = lines + line
                            if i == 5:
                                pbar.update(len(lines.encode('utf-8')))
                                # pbar.display()
                                i = 0
                                if in_memory:
                                    sentence_id = self.insert_memory_script(lines, sentence_id, doc_id, g)
                                else:
                                    sentence_id = self.insert_db_script(lines, sentence_id, doc_id)
                                sentence_id = sentence_id + 1
                                lines = ''
                            i += 1
                        if lines:
                            if in_memory:
                                sentence_id = self.insert_memory_script(lines, sentence_id, doc_id, g)
                            else:
                                sentence_id = self.insert_db_script(lines, sentence_id, doc_id)
            doc_id += 1
        if in_memory:
            g.serialize(destination=save_file+".owl", format="xml")
