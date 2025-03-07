FILE_CONVERSION_PROMPT = """
-Target activity-
You are an intelligent assistant that helps a human convert a file with a certain structure into a file with a different structure.

-Goal-
Given a sentence with a list of sentences, predicates, subject, object, extra and date convert each line of the document into the output shown in the examples

-Steps-
1. Identify which words in the sentence make up the predicate(PREDICATE), object(OBJECT) and subject(SUBJECT)
2. Re-write the sentence applying the correct tags on each word that belongs to predicate (PREDICATE), object(OBJECT) or subject(SUBJECT)
3. Format each sentence in the following format
word_1 word_2|SUBJECT word_3|SUBJECT word_4 word_5|PREDICATE word_6 word_7|OBJECT
4. Do not add any explanation

Example 1:

SUBJECT: 32.7 % of all households
PREDICATE: were made up of
OBJECT: individuals
Original: 32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older .

Output: 32.7|SUBJECT %|SUBJECT of|SUBJECT all|SUBJECT households|SUBJECT  were|PREDICATE made|PREDICATE up|PREDICATE of|PREDICATE  individuals|OBJECT  and 15.7 % had someone living alone who was 65 years of age or older .

Example 2:
SUBJECT: 15.7 % of all households
PREDICATE: had
OBJECT: someone living alone who was 65 years of age or older
Original: 32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older .

Output: 32.7 % of|SUBJECT all|SUBJECT households|SUBJECT were made up of individuals and 15.7|SUBJECT %|SUBJECT had|PREDICATE  someone|OBJECT living|OBJECT alone|OBJECT who|OBJECT was|OBJECT 65|OBJECT years|OBJECT of|OBJECT age|OBJECT or|OBJECT older|OBJECT  .

Example 3:
SUBJECT: A `` prime '' manifold , a connected sum of more than one manifold
PREDICATE: is not the
OBJECT: sphere of the same dimension,,,
Original: A manifold is `` prime '' if it can not be presented as a connected sum of more than one manifold , none of which is the sphere of the same dimension ."

Output: A|SUBJECT manifold|SUBJECT is|PREDICATE ``|SUBJECT prime|SUBJECT ''|SUBJECT if it can not be presented as a|SUBJECT connected|SUBJECT sum|SUBJECT of|SUBJECT more|SUBJECT than|SUBJECT one|SUBJECT manifold|SUBJECT , none of which is the sphere|OBJECT of|OBJECT the|OBJECT same|OBJECT dimension|OBJECT .

-Real Data-
SUBJECT: {a1}
PREDICATE: {r}
OBJECT: {a2}
Original: {sentence}
Output: """