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

def test_open():
    #Open the test file
    f='./tests/test.txt.redacted'
    document2 = open(f).read()
    assert document2 is not None

def test_blanks():

    #Open the test file
    f='./tests/test.txt.redacted'
    document2 = open(f).read()
    print("Now opening redacted document: ", f)

    nlp = spacy.load('en')
    #Parse the file
    document=nlp(document2)

    #Set a regex pattern for a number of blocks, one space, and then some number of blocks
    NUM_PATTERN = re.compile(r"[_█]{2,}[ ][_█]{2,}")
    redactions=[]

    #Search the file for redactions and store in a dataframe.
    #The dataframe has three keys: the pattern of text, its starting location, and
    #its ending location
    i=0
    for match in re.finditer(NUM_PATTERN, document.text):
        start, end = match.span()
        print(f"Redaction found: '{document.text[start:end]}'")
        span = document.char_span(start, end)
        text=document.text[start:end]
        df_tmp = pd.DataFrame({'text':[text], 'start':[start], 'end':[end]})
        i+=1
        if i==1:
            df2=df_tmp
        else:
            frames = [df2, df_tmp]
            df2 = pd.concat(frames,ignore_index=True)
            df2.reset_index(drop=True)
    #Split the redaction blocks, then count their length and store them as
    #keys in the df
    new=df2['text'].str.split(" ",n=1,expand=True)
    df2['first']=new[0]
    df2['last']=new[1]
    df2['first.c']=df2['first'].str.len()
    df2['last.c']=df2['last'].str.len()
#*****************************************************
#Assert that the character which has been identified as a redaction block
#is in fact a redaction block character
#*****************************************************
    assert df2['first'][0][0]=='█'

