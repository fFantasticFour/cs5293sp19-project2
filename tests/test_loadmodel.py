import spacy
import nltk
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from gensim.models.keyedvectors import KeyedVectors
import glob
import pandas as pd
import re
from tqdm import tqdm
#you may need to download the stopwords list below:
#if so, uncomment the line below
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string

def test_loadmodel():
    # load saved model
    model = Word2Vec.load('celeb_word2vec_model')

    # load saved word vectors
    word_vectors = KeyedVectors.load('word_vectors')
#assert that the most similar bigram in the model to 'kung_fu' is 'martial_arts'
    assert model.wv.most_similar('kung_fu',topn=1)[0][0] == 'martial_arts'


