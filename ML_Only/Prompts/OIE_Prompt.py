# Prompt definition for OIE LLM Querying

OIE_EXTRACTION_PROMPT = """
-Target activity-
You are an intelligent assistant that helps a human analyst to analyse Sentences and extract relationships from them.

-Goal-
Given a sentence that is potentially relevant to this activity, a list of potential subject, predicate and objects extract all relationships present in the sentece.

-Steps-
1. Given the list of predicates identify the most relevant ones to become a relationship.

2. For each predicate identified in step 1, extract all the possible relationships in the given sentence.
For each predicate, extract the following information:
- Subject: name of the entity that is subject of the predicate. The subject entity is one that committed the action.
- Object: name of the entity that is object of the predicate. The object entity is one that either reports/handles or is affected by the action described in the predicate.
If a triple has been previously identified you can skip it.

3. Remove any triples DUPLICATE triples identified. A DUPLICATE triple is one that contains the same information as another triple.

4. Return output as a single list of all the triples identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. Attribute a Certainty rating: a float score between 0-10 that represents the RELEVANCE of the Triple.  RELEVANCE is the importance of triple to understand the sentence.

Format each triple as {subject_delimiter}<subject_entity>{predicate_delimiter}<predicate_entity>{object_delimiter}<object_entity>{certainty_delimiter}<certainty_rating>

6. When finished, output {completion_delimiter}. DO NOT add any Explanation.

-Examples-
Example 1:
Subject Candidates: Nolan Bushnell / AgingGames
Predicate Candidates: Nolan Bushnell está / está no conselho consultivo da Anti - AgingGames
Object Candidates: no / conselho consultivo / no consultivo / consultivo da Anti - AgingGames
Sentence: Nolan Bushnell esta no conselho consultivo da Anti - AgingGames .
Output:
{subject_delimiter}Nolan Bushnell{predicate_delimiter}está no{object_delimiter}consultivo da Anti - AgingGames{certainty_delimiter}10
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Subject Candidates: {subject_candidates}
Predicate Candidates: {predicate_candidates}
Object Candidates: {object_candidates}
Sentence: {input_text}
Output:"""

TRIPLE_FILTER_PROMPT = """
-Target activity-
You are an intelligent assistant that helps a human analyst to analyse Sentences and extract relationships from them.

-Goal-
Given a sentence and a list of potential triples (subject / predicate / object) extracted from it remove duplicate triples keeping only unique ones.

-Steps-
1. Identify which triples are repeated. A repeated triple is one that contains the same information as another triple or the same Subject, Predicate and Object.

2. Return output as a single list of all the triples identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. Format each triple as {subject_delimiter}<subject_entity>{predicate_delimiter}<predicate_entity>{object_delimiter}<object_entity>{certainty_delimiter}<certainty_rating>

4. When finished, output {completion_delimiter}. DO NOT add any Explanation.

-Examples-
Example 1:
Triple 1: Nolan Bushnell / está no / consultivo da Anti - AgingGames
Triple 2: Nolan Bushnell / está no / conselho consultivo da Anti - AgingGames
Sentence: Nolan Bushnell esta no conselho consultivo da Anti - AgingGames .
Output:
{subject_delimiter}Nolan Bushnell{predicate_delimiter}está no{object_delimiter}conselho consultivo da Anti - AgingGames{certainty_delimiter}10
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Sentence: {input_text}
"""

# CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
# LOOP_PROMPT = "It appears some entities and relationships may have still been missed.  Answer YES | NO if there are still entities or relationships that need to be added.\n"