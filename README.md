# cs5293sp19-project2

This program attempts to fill texts whose names have been redacted with the name most appropriate for that space.

Its strategy is NOT based on classifiers. Instead, it is based on word vectors.

In my previous project, the concept of word vectors very much interested me. So, in this project, I felt like
if you created a word vector with the words surrounding the redacted name, it would contain enough of
the connotative content of that name to find the correct replacement.

My training set was the positive and negative reviews in the imdb database for what they called both test and
train. This results in around 50,000 reviews. My test set is the reviews in the imdb database marked 'unlabeled'.
This gives me 50000 more review to pull from, though in practice I provide only 500 in the redacted_docs
directory.

The analysis proceeded in three steps:

1) The off-the-shelf model and dictionary of word vectors in Spacy has connotations for names (like ashton),
but not for particular celebrities (like ashton_kutcher). So, first, I created a list of bigrams.

For this, I used the gensim module. I referred to the article 'Celebrity Word Vectors' at
http://engineering.pmc.com/2017/08/celebrity-word-vectors-79/
to build my model. However, his dataset was articles in the hollywoodlife database,
whereas I substituted reviews from the imdb database.

Using the gensim modules Phrases and Phraser, I created a list of bigrams which appear at
least twice in the training set, with a threshold of 2.

2) Then, using word2vec, I created a model which represents these bigrams as vectors. As a result,
the word vector for say 'nicolas_cage' is now a vector of length 100 which can be compared to
other celebrities, or to other sentence vectors.
 
The gensim list of bigrams and word2vec model are saved to this directory, and loaded by the program.

3) My list of bigrams (and resulting word vectors) contains many more bigrams than just celebrity names.
For example, 'action_movie' was a very common bigram. In order to tell if a particular bigram was a name, 
I needed a database of celebrity names.

For this, I used the imdb rank of the top 2 million movie stars. It is held at
https://datasets.imdbws.com/

The edited version of this file (20,000 names) is saved as 'film_bigrams.tsv' and is loaded by the program.

Once I had a model for each celebrity in the imdb database, I could then find the similarity between the redaction's context and
the potential names which could replace it.

Most of the code in main.py is just data frame manipulation to arrive at a list of words surrounding a redacted name, called
the sentence_stream. These words give the connotation of the sentence surrounding the missing name. For example, if the 
sentence contains the words 'james_bond', then the model gives

model.wv.most_similar(positive=['james_bond'])

gives

[('roger_moore', 0.8767318725585938),
 ('bond', 0.8473519086837769),
 ('sean_connery', 0.8351479768753052),
 ('octopussy', 0.8349493741989136),
 ('klaus_maria', 0.8182209730148315),
 ('goldeneye', 0.8171074390411377),
 ('barbara_carrera', 0.8135736584663391),
 ('bond_films', 0.7988383769989014),
 ('thunderball', 0.7979954481124878),
 ('connery', 0.7971374988555908)]

Or, if the sentence contains kung_fu, this gives

model.wv.most_similar('kung_fu')

[('martial_arts', 0.9202347993850708),
 ('shaolin_temple', 0.8921620845794678),
 ('kungfu', 0.8698007464408875),
 ('jet_li', 0.8665159940719604),
 ('jackie_chan', 0.8597754240036011),
 ('black_mask', 0.8563443422317505),
 ('martial_art', 0.8511183261871338),
 ('hong_kong', 0.847243070602417),
 ('bruce_lee', 0.8326942920684814),
 ('sammo', 0.8201808929443359)]

Only one of these bigrams is likely to fit the redacted name, solving the unredaction problem.

To arrive at a ranking of possible names to fit the redaction, I find all of the words surrounding the missing name. Each of these words
carries a connotative meaning represented by its word vector.
The word vectors for these words are added together and compared to each of the possible bigram candidates,
using the command

rank = [model.n_similarity(sentence_stream2,[i]) for i in candidate_bigrams if i in model.wv.vocab]

The 5 names which are most similar to their context are output as the solution to this problem.

WHAT WORKS
If the context immediately surrounding a redacted name has some identifying content in it, the algorithm
gives a correct result. For example, 'I blame ______ ___ for letting this project move forward' correctly
identifies michael_bay. Contexts that include 'james bond' or 'martial arts' also find their correct name. 

WHAT DOESN'T
If the content surrounding a redacted name contains none of the words most similar to the bigram you're trying to
find, then the method fails. It is still possible to find the correct match, but if the length of the names
are very common (like 6 or 7 letters), the algorithm has too many names to choose from.

*************************************************************************************************
Usage

python3 project2/main.py --input 'docs/test.txt.redacted'

where the file 'test.txt.redacted' is the file to be redacted. You may use any of the redacted files in the directory
/redacted_docs.
*************************************************************************************************

Testing

The file test_names tests the redaction process used to create the training files. The sentence
'My name is Brandon'
results in 'Brandon', as expected.

The file test_findblanks tests the regex search for redaction spaces. It adds the redaction blocks to a dataframe,
then confirms that the characters it has selected from the text are in fact blocks.

The file test_loadmodel tests the word vector model. It loads the model into memory, then confirms that
the word vector 'martial_arts' is the one most similar to 'kung_fu'


