import os
import string
from time import sleep
import warnings
from SPARQLWrapper import SPARQLWrapper, POST
from spacy_conll import init_parser
from tqdm import tqdm
from unidecode import unidecode
from Query_Builder import QueryBuilder
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF, DOAP, FOAF, ORG, OWL, SKOS, XSD
import re

class CreateGraph:
    """
    Main class used to create the knowledge graphs.

    """
    def __init__(self, folder, graph_name, extra_connetions = [], main_uri='http://ieeta.pt/ontoud#',
                    connection_string='http://localhost:8890/sparql', language="pt_core_news_sm", preprocessing = None, in_memory = False):
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
        self.existing_words = []
        if not in_memory:
            self.connection = connection_string
            self.sparql = SPARQLWrapper(self.connection)
            self.sparql.setMethod(POST)
            self.queries = QueryBuilder(self.main_uri, self.graph_name)
        else:
            self.g = Graph()
            self.insert_binds()

    def insert_binds(self):
        #Inserts some common prefixes
        self.g.bind("rdfs", RDFS)
        self.g.bind("rdf", RDF)
        self.g.bind("doap", DOAP)
        self.g.bind("org", ORG)
        self.g.bind("owl", OWL)
        self.g.bind("skos", SKOS)
        self.g.bind("xsd", XSD)
        self.g.bind("foaf", FOAF)

    def insert_classes(self):
        """
        3 classes: Words, Sentences, Documents
        :return:
        """
        self.g.add((URIRef(self.main_uri+"Documents"), RDF.type, OWL.Class))
        self.g.add((URIRef(self.main_uri+"Sentences"), RDF.type, OWL.Class))
        self.g.add((URIRef(self.main_uri+"Words"), RDF.type, OWL.Class))


    def insert_object_properties(self):
        """
        Object propreties:
        From document to sentences:
            document - containsSentence - sentence
        From sentence to document:
            sentence - fromDocument - document
        From sentence to sentence:
            sentence - nextSentence - sentence
            sentence - previousSentence - sentence
        From sentence to word:
            sentence - depGraph - word (the root word of a sentence, where the dependency graph begins)
        From word to sentence:
            word - fromSentence - sentence
        From word to word:
            word - depGraph - word (the dependency of a word on another word)
        """
        #Document
        self.g.add((URIRef(self.main_uri + "containsSentence"), RDF.type, OWL.ObjectProperty))
        #Sentence
        self.g.add((URIRef(self.main_uri + "fromDocument"), RDF.type, OWL.ObjectProperty))
        self.g.add((URIRef(self.main_uri + "nextSentence"), RDF.type, OWL.ObjectProperty))
        self.g.add((URIRef(self.main_uri + "previousSentence"), RDF.type, OWL.ObjectProperty))
        #Word
        self.g.add((URIRef(self.main_uri + "depGraph"), RDF.type, OWL.ObjectProperty))
        self.g.add((URIRef(self.main_uri + "fromSentence"), RDF.type, OWL.ObjectProperty))
        self.g.add((URIRef(self.main_uri + "depGraph"), RDF.type, OWL.ObjectProperty))

    def insert_data_properties(self):
        """
        Data Properties:
        Documents:
            Nothing
        Sentence:
            Text: full text of the sentence
        Words:
            edge: property from the syntactic parser
            feats: there are several feats, each separated into its own property, from the syntactic parser.
            pos: property from the syntactic parser
            poscoarse: property from the syntactic parser, on same languages is equal to the pos tag.
            originalWord: since we are storing lemma this is the original word.
            entityType: the type of entity in our text.
        """
        #Sentence
        self.g.add((URIRef(self.main_uri + "Text"), RDF.type, OWL.DatatypeProperty))
        #Word
        self.g.add((URIRef(self.main_uri + "Edge"), RDF.type, OWL.DatatypeProperty))
        self.g.add((URIRef(self.main_uri + "Pos"), RDF.type, OWL.DatatypeProperty))
        self.g.add((URIRef(self.main_uri + "PosCoarse"), RDF.type, OWL.DatatypeProperty))
        self.g.add((URIRef(self.main_uri + "OriginalWord"), RDF.type, OWL.DatatypeProperty))
        self.g.add((URIRef(self.main_uri + "entityType"), RDF.type, OWL.DatatypeProperty))

    def create_graph(self, fileName="MyGraph"):
        self.insert_classes()
        self.insert_object_properties()
        self.insert_data_properties()

        doc_id = 1
        i = 0
        lines = ''
        files = [f for f in os.listdir(self.folder_name) if os.path.isfile(os.path.join(self.folder_name, f))]
        for file_name in files:
            if not file_name.startswith("."):
                sentence_id = 1
                file_path = os.getcwd() + "/" + self.folder_name + "/" + file_name
                print(f"--- Processing file {doc_id} : {file_name} ---")
                with tqdm(total=os.path.getsize(file_path)) as pbar:
                    with open(file_path) as file:
                        for line in file:
                            text_nohtml = re.sub(r'http\S+', '', line)
                            text_nohtml = text_nohtml.lower()
                            if self.preprocessing:
                                processed_lines = self.preprocessing(text_nohtml)
                                sentence = ""
                                for line in processed_lines:
                                    sentence += line[0] + " "
                            else:
                                sentence = lines
                            sentence = re.sub(r'\s([?.!:\-,"](?:\s|$))', r'\1', sentence)
                            sentence = sentence.replace("  ", " ")
                            sentence = sentence.strip()
                            print(sentence)