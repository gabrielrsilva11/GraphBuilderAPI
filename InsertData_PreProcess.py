import os
import string
from time import sleep
import warnings
from SPARQLWrapper import SPARQLWrapper, POST
#from spacy_conll import init_parser
from tqdm import tqdm
from unidecode import unidecode
from Query_Builder import QueryBuilder
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF, DOAP, FOAF, ORG, OWL, SKOS, XSD
from wikimapper import WikiMapper
import re
from fastcoref import spacy_component
import spacy
import logging

logging.getLogger("fastcoref").setLevel(logging.WARNING)
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
        # self.nlp = init_parser(language,
        #                        "spacy",
        #                        ext_names={"conll_pd": "pandas"},
        #                        #disable_sbd=True,
        #                        #parser_opts={"use_gpu": True, "verbose": False},
        #                        include_headers=True)
        self.nlp = spacy.load("en_core_web_trf")
        config = {"ext_names": {"conll_pd": "pandas"}}
        self.nlp.add_pipe("conll_formatter", config=config, last=True)
        self.nlp.add_pipe("fastcoref")
                     #config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref'})
        self.connection = connection_string
        self.sparql = SPARQLWrapper(self.connection)
        self.mapper = WikiMapper("Data_to_process/wikimapper_data/en")
        #self.sparql.setCredentials("dba", "dbapass")
        self.sparql.setMethod(POST)
        self.queries = QueryBuilder(self.main_uri, self.graph_name)
        self.g = Graph()
        # Document navigation -> Text, Sentence, Word classes
        self.c_text_uri = self.main_uri + "Text"
        self.c_sentence_uri = self.main_uri + "Sentence"
        self.c_word_uri = self.main_uri + "Word"
        self.c_attributes_uri = self.main_uri + "Attributes"

        # General properties -> text/conll properties to be created as object properties in the graph
        self.o_depgraph_uri = self.main_uri + "depGraph"
        self.o_nextsentence_uri = self.main_uri + "nextSentence"
        self.o_previoussentence_uri = self.main_uri + "previousSentence"
        self.o_nextword_uri = self.main_uri + "nextWord"
        self.o_previousword_uri = self.main_uri + "previousWord"
        self.o_contains_sentence = self.main_uri + "containsSentence"
        self.o_from_text = self.main_uri + "fromText"
        self.o_contains_text = self.main_uri + "containsText"
        self.o_from_sentence_uri = self.main_uri + "fromSentence"
        self.o_coreference_uri = self.main_uri + "coReference"
        self.o_edge_uri = self.main_uri + "edge"
        self.o_pos_uri = self.main_uri + "pos"
        self.o_poscoarse_uri = self.main_uri + "poscoarse"
        self.o_feats_uri = self.main_uri + "feats"
        #Extra object properties
        self.extra_object_properties = self.fetch_extra_properties(extra_connetions)

        # CoNLL properties -> EDGE, FEATS, ID, LEMMA, POS, POS_COARSE, WORD as a data property
        self.o_head_uri = self.main_uri + "head"
        self.d_sentence_text = self.main_uri + "senttext"

        #self.d_feats_uri = self.main_uri + "feats"
        self.d_id_uri = self.main_uri + "id"
        self.d_lemma_uri = self.main_uri + "lemma"
        self.d_word_uri = self.main_uri + "word"
        self.d_wikimapper_uri = self.main_uri + "wikidataId"
        #dict to keep track of the already inserted feats
        self.d_feats_list = []
        self.feats_specific_list = []
        self.objectDict = {"edge": [], "pos": [], "poscoarse": []}
        # self.edges_list = []
        # self.pos_list = []
        # self.poscoarse_list = []

    def fetch_extra_properties(self, extra_connetions):
        connections_list = []
        for name in extra_connetions:
            connections_list.append(self.main_uri+name)
        print(connections_list)
        return connections_list

    def insert_relationship_data(self):
        if self.in_memory:
            self.g.bind("rdfs", RDFS)
            self.g.bind("rdf", RDF)
            self.g.bind("doap", DOAP)
            self.g.bind("org", ORG)
            self.g.bind("owl", OWL)
            self.g.bind("skos", SKOS)
            self.g.bind("xsd", XSD)
            self.g.bind("foaf", FOAF)

        self.insert_data(self.c_text_uri, RDF.type, OWL.Class)
        self.insert_data(self.c_sentence_uri, RDF.type, OWL.Class)
        self.insert_data(self.c_word_uri, RDF.type, OWL.Class)
        self.insert_data(self.c_attributes_uri, RDF.type, OWL.Class)

        self.insert_data(self.o_pos_uri, RDFS.subClassOf, self.c_attributes_uri)
        self.insert_data(self.o_poscoarse_uri, RDFS.subClassOf, self.c_attributes_uri)
        self.insert_data(self.o_edge_uri, RDFS.subClassOf, self.c_attributes_uri)
        self.insert_data(self.o_feats_uri, RDFS.subClassOf, self.c_attributes_uri)

        # object properties
        self.insert_data(self.o_head_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_depgraph_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_nextsentence_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_previoussentence_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_nextword_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_previousword_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_contains_text, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_contains_sentence, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_from_text, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_previousword_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_from_sentence_uri, RDF.type, OWL.ObjectProperty)
        self.insert_data(self.o_coreference_uri, RDF.type, OWL.ObjectProperty)

        # Insert the extras
        if self.extra_object_properties:
            for extra_object in self.extra_object_properties:
                self.insert_data(extra_object, RDF.type, OWL.DatatypeProperty)

        # data properties
        self.insert_data(self.d_sentence_text, RDF.type, OWL.DatatypeProperty)
        self.insert_data(self.d_id_uri, RDF.type, OWL.DatatypeProperty)
        self.insert_data(self.d_lemma_uri, RDF.type, OWL.DatatypeProperty)
        self.insert_data(self.d_word_uri, RDF.type, OWL.DatatypeProperty)
        self.insert_data(self.d_wikimapper_uri, RDF.type, OWL.DatatypeProperty)

    def insert_data(self, s, p, o):
        """
        Inserts a triple into a triple-storage and a knowledge graph.

        :param s: subject of the triple
        :param p: predicate of the triple
        :param o: object of the triple
        :return:
        """
        if self.in_memory:
            if type(s) is not URIRef:
                s = URIRef(s)
            if type(p) is not URIRef:
                p = URIRef(p)
            if type(o) is not URIRef and type(o) is not Literal:
                o = URIRef(o)
            self.g.add((s, p, o))
        else:
            query = self.queries.build_insert_query(s, p, o)
            self.sparql.setQuery(query)
            # wrapper.method = 'POST'
            for i in range(0, 10):
                try:
                    results = self.sparql.query()
                    str_error = None
                except:
                    str_error = "Error"
                    pass

                if str_error:
                    if i == 0:
                        print(query)
                    print("Error occurred. Attempting to upload triple again.")
                    print("Attempt number: ", i)
                    sleep(2*i)  # wait for 2*attempt number seconds before trying to fetch the data again
                else:
                    break

    def insert_script(self, lines, sentence_id, doc_id, coref):
        """
        Main script to build the graph
        :param lines: the text to insert
        :param sentence_id: last known sentence_id for identification purposes.
        :param doc_id: the id of the document we are currently processing
        :return: the last used sentence_id.
        """
        text = ""
        text_nohtml = re.sub(r'http\S+', '', lines)
        #text_nohtml = text_nohtml.lower()
        if self.preprocessing:
            processed_lines = self.preprocessing(text_nohtml)
            sentence = ""
            for line in processed_lines:
                sentence += line[0] + " "
        else:
            sentence = lines

        sentence_complete = sentence.strip()
        doc = self.nlp(sentence_complete)
        text = text+sentence_complete
        conll = doc._.pandas
        sentence = []
        boundaries = {}
        first_bound = 0
        lower_bound = 0
        upper_bound = 0
        word_boundaries = {}
        textid_uri = URIRef(self.c_text_uri + "_" + str(doc_id))
        self.insert_data(textid_uri, RDF.type, self.c_text_uri)
        indexes_used = []
        for index, row in conll.iterrows():
            word = row['FORM'].replace("'", "").replace("\"", "")
            lemma = row['LEMMA'].replace("'", "").replace("\"", "")
            word_id = row['ID']
            sentence.append(unidecode(word))
            if row['ID'] == 1:
                sentence_id+=1
                sentenceid_uri = self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id)
                # if sentence_id > 0:
                #     sentence = [sentence[-1]]
                #     self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri)
                #     self.insert_data(sentenceid_uri, self.o_from_text, textid_uri)
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(sentenceid_uri, RDF.type, self.c_sentence_uri)
                self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri)
                self.insert_data(sentenceid_uri, self.o_from_text, textid_uri)
                if sentence_id != 1:
                    boundaries[sentence_id-1] = [[first_bound, upper_bound], ' '.join(sentence[0:-1]).strip(), word_boundaries]
                    word_boundaries = {}
                    for c in word:
                        upper_bound += 1
                    word_boundaries[word_id] = [word, lower_bound, upper_bound]
                    # account for spaces
                    upper_bound += 1
                    first_bound = upper_bound
                    lower_bound = upper_bound
                    #Sent Text
                    self.insert_data(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1),
                                     self.d_sentence_text, Literal(' '.join(sentence[0:-1]).strip()))
                    sentence = []
                    sentence.append(unidecode(word))
                    #Previous sentence
                    self.insert_data(sentenceid_uri, self.o_previoussentence_uri,
                                     self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1))
                    #Next sentence
                    self.insert_data(self.c_sentence_uri + "_" + str(doc_id) + "_" + str(sentence_id - 1), self.o_nextsentence_uri,
                                     sentenceid_uri)
                else:
                    for c in word:
                        upper_bound += 1
                    word_boundaries[word_id] = [word, lower_bound, upper_bound]
                    # account for spaces
                    upper_bound += 1
                    lower_bound = upper_bound
            else:
                word_id = row['ID']
                previous_uri = wordid_uri
                wordid_uri = self.d_word_uri + "_" + str(doc_id) + "_" + str(sentence_id) + "_" + str(word_id)
                self.insert_data(wordid_uri, self.o_previousword_uri, previous_uri)
                self.insert_data(previous_uri, self.o_nextword_uri, wordid_uri)
                for c in word:
                    upper_bound += 1
                word_boundaries[word_id] = [word, lower_bound, upper_bound]
                # account for spaces
                upper_bound += 1
                lower_bound = upper_bound

            self.insert_data(wordid_uri, RDF.type, self.c_word_uri)
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['ID']))
            self.insert_data(wordid_uri, self.d_word_uri, Literal(word))
            self.process_feats(wordid_uri, row['FEATS'])
            #self.insert_data(wordid_uri, self.d_feats_uri, Literal(row['feats']))
            self.insert_data(wordid_uri, self.d_id_uri, Literal(row['ID']))
            self.insert_data(wordid_uri, self.d_lemma_uri, Literal(lemma))
            self.process_conll_as_objects('edge', wordid_uri, self.o_edge_uri, row['DEPREL'])
            self.process_conll_as_objects('pos', wordid_uri, self.o_pos_uri, row['UPOS'])
            self.process_conll_as_objects('poscoarse', wordid_uri, self.o_poscoarse_uri, row['XPOS'])
            self.insert_wikimapper(wordid_uri, word)
            if self.preprocessing:
                for o in range(0, len(processed_lines)):
                    word_to_check = processed_lines[o][0].strip().lower()
                    if word.lower() in word_to_check and o <= index and o not in indexes_used:
                        indexes_used.append(o)
                        for k in range(0, len(self.extra_object_properties)):
                            if processed_lines[o][k + 1] != '':
                                self.insert_data(wordid_uri, URIRef(self.extra_object_properties[k]),
                                       Literal(processed_lines[o][k + 1]))
                        break

            if row['HEAD'] == 0:
                # print(sentence)
                self.insert_data(wordid_uri, self.o_from_sentence_uri, sentenceid_uri)
                self.insert_data(sentenceid_uri, self.o_depgraph_uri, wordid_uri)
            else:
                self.insert_data(wordid_uri, self.o_head_uri,
                                 self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['HEAD']))
                self.insert_data(self.d_word_uri + "_" + str(doc_id) +"_" + str(sentence_id) + "_" + str(row['HEAD']), self.o_depgraph_uri,
                                 wordid_uri)
        self.insert_data(sentenceid_uri, self.d_sentence_text, Literal(' '.join(sentence).strip()))
        self.insert_data(textid_uri, self.o_contains_sentence, sentenceid_uri)
        self.insert_data(sentenceid_uri, self.o_from_text, textid_uri)
        if coref:
            self.insert_coreference(doc, doc_id, boundaries, text)
        return sentence_id

    def process_feats(self, wordid_uri, feats, g = None):
        split_feats = feats.split("|")
        if split_feats[0] != "_":
            for feat in split_feats:
                feat = feat.split("=")
                feat[1] = feat[1].replace(",", "_")
                if feat[0] in self.d_feats_list:
                    if feat[1] in self.feats_specific_list:
                        self.insert_data(wordid_uri, self.main_uri+feat[0].lower(), self.main_uri+feat[1].lower())
                    else:
                        self.feats_specific_list.append(feat[1])
                        self.insert_data(self.main_uri + feat[1].lower(), RDF.type, self.main_uri+ feat[0].lower())
                        self.insert_data(wordid_uri, self.main_uri+feat[0].lower(), self.main_uri+feat[1].lower())
                else:
                    self.d_feats_list.append(feat[0])
                    self.feats_specific_list.append(feat[1])
                    self.insert_data(self.main_uri+feat[0].lower(), RDFS.subClassOf, self.o_feats_uri)
                    self.insert_data(self.main_uri+feat[0].lower(), RDF.type, OWL.ObjectProperty)
                    self.insert_data(self.main_uri + feat[1].lower(), RDF.type, self.main_uri+feat[0].lower())
                    self.insert_data(wordid_uri, self.main_uri+ feat[0].lower(), self.main_uri +feat[1].lower())

    def process_conll_as_objects(self, prop_type, word, uri, to_add):
        if prop_type == "poscoarse":
            transformed = ''
            for character in to_add:
                if character in string.punctuation:
                    transformed = transformed+str(ord(character))
                else:
                    transformed = transformed+character
            to_add = transformed

        to_add_uri = self.main_uri + to_add
        added = True
        if to_add not in self.objectDict[prop_type]:
            self.objectDict[prop_type].append(to_add)
            added = True
        else:
            added = False
        if added:
            self.insert_data(uri, RDF.type, OWL.ObjectProperty)
            self.insert_data(to_add_uri, RDF.type, uri)
        self.insert_data(word, uri, to_add_uri)

    def insert_wikimapper(self, word_id, word):
        wiki_id = self.mapper.title_to_id(word)
        if wiki_id:
            self.insert_data(word_id, self.d_wikimapper_uri, Literal(wiki_id))

    def fetch_root_word(self, text):
        first_ref_process = self.nlp(text)
        root_first_ref = first_ref_process._.pandas
        root_word = root_first_ref['FORM'].loc[root_first_ref['DEPREL'] == 'ROOT']
        if len(root_word.values) > 0:
            return root_word.values[0]
        else:
            return None

    def insert_coreference(self, doc, doc_id, boundaries, text):
        #Go through all the co-reference clusters
        #Fetch the root cluster (first reference) and

        for coref_cluster in doc._.coref_clusters:
            ids_used = []
            first_ref = coref_cluster[0]
            first_ref_text = text[first_ref[0]:first_ref[1]]
            root_word_id = ''
            #Fetch the root word in case its an expression
            if len(first_ref_text.split(" ")) > 1:
                root_word = self.fetch_root_word(first_ref_text)
            else:
                root_word = first_ref_text

            #Fetch the sentence and word ID of the root word
            for key, item in boundaries.items():
                if boundaries[key][0][0] <= first_ref[0] <= boundaries[key][0][1]:
                    first_ref_sentence_id = key
                    word_id = 1
                    for key2, item2 in boundaries[key][2].items():
                        if item2[0] == root_word:
                            root_word_id = word_id
                            break
                        else:
                            word_id += 1
                    break
            if root_word_id != '':
                ids_used.append(str(first_ref_sentence_id)+"_"+str(root_word_id))
                root_sentence_id = self.d_word_uri + "_" + str(doc_id) + "_" + str(first_ref_sentence_id) + "_" + str(root_word_id)
            #    print(root_sentence_id)
            #     #MATCH ROOT WORD TO SENTENCE AND WORD
            #
                for ref in coref_cluster[1:]:
                    second_ref_text = text[ref[0]:ref[1]]
                    second_ref_text_split = second_ref_text.split(" ")
                    if len(second_ref_text_split) > 1:
                        reference_word = self.fetch_root_word(second_ref_text)
                    else:
                        reference_word = second_ref_text
                    # Fetch the sentence and word ID of the root word
                    ref_word_id = ''
                    for key, item in boundaries.items():
                        if boundaries[key][0][0] <= ref[0] <= boundaries[key][0][1]:
                            second_ref_sentence_id = key
                            word_id = 1
                            for key2, item2 in boundaries[key][2].items():
                                sentence_word_id = str(second_ref_sentence_id) + "_" + str(word_id)
                                if item2[0] == reference_word and sentence_word_id not in ids_used:
                                    ref_word_id = word_id
                                    ids_used.append(str(first_ref_sentence_id) + "_" + str(root_word_id))
                                    break
                                else:
                                    word_id += 1
                            break
                    if ref_word_id != '':
                        reference_sentence_id = self.d_word_uri + "_" + str(doc_id) + "_" + str(second_ref_sentence_id) + "_" + str(
                            ref_word_id)
                        self.insert_data(root_sentence_id, self.o_coreference_uri, reference_sentence_id)
        return

    def create_graph(self, save_file = "Serialized", coref = False):
        """
        :param in_memory: Boolean which indicates whether we want to create the graph in-memory or upload to a storage.
        :param save_file: Name of the file to save the graph.
        """
        doc_id = 0
        i = 0
        lines = ''
        files = [f for f in os.listdir(self.folder_name) if os.path.isfile(os.path.join(self.folder_name, f))]
        self.insert_relationship_data()
        for file_name in files:
            if not file_name.startswith("."):
                sentence_id = 0
                file_path = os.getcwd()+"/"+self.folder_name+"/"+file_name
                print(f"--- Processing file {doc_id} : {file_name} ---")
                with tqdm(total=os.path.getsize(file_path)) as pbar:
                    with open(file_path) as file:
                        for line in file:
                            #print(repr(line))
                            if line != "\n":
                                lines = lines + line + " "
                                i += 1
                                if i == 50:
                                    sentence_id = self.insert_script(lines, sentence_id, doc_id, coref)
                                    sentence_id = sentence_id + 1
                                    pbar.update(len(lines.encode('utf-8')))
                                    # pbar.display()
                                    i = 0
                                    lines = ''
                        pbar.update(len(lines.encode('utf-8')))
                    # if lines:
                    #         sentence_id = self.insert_db_script(lines, sentence_id, doc_id)
            doc_id += 1
        if self.in_memory:
            self.g.serialize(destination=save_file+".ttl", format="turtle")
