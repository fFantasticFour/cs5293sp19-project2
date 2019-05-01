import argparse
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

#a function used to eliminate the redaction blocks
def replace_block(word):
    word2=''
    for c in word:
        if (c == '█'):
            c = ''
        word2+=c
    return word2

def main(args):

    #load in a bunch of data:
    #The spacy natural language processor for english
    nlp = spacy.load('en')

    #A list of names from the imdb database, parsed into bigrams
    #and listed with a key 'wv.exists' indicating if the movie
    #star is in the model.
    film_bigrams=pd.read_csv('film_bigrams.tsv',sep='\t')
    #Only keep a name if it is in the model
    film_bigrams=film_bigrams[film_bigrams['wv.exists']==True]

    # load saved bigram model of celebrity names
    model = Word2Vec.load('celeb_word2vec_model')

    # load saved word vectors for celebrity names
    word_vectors = KeyedVectors.load('word_vectors')

#    for f in glob.glob(args.input):
    #Open the redacted file
#    f='./redacted_docs/39337_0.txt.redacted'
    for f in glob.glob(args.input):
     #   f = args.input
        document2 = open(f).read()
        print("Now opening redacted document: ", f)

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

        for blank_iter in range(0,df2.shape[0]):
            #Search the celebrity database for names which match the redaction pattern
            #(i.e their first and last names fit the length of the blanks)
            first_blank=df2['first.c'][blank_iter]
            last_blank=df2['last.c'][blank_iter]
            candidates=film_bigrams[(film_bigrams['first.c']==first_blank) & (film_bigrams['last.c']==last_blank)]
            candidate_bigrams=candidates['bigram']
            #Make a list of the possible candidate celebrities
            candidate_bigrams=candidate_bigrams.reset_index(drop=True)

            #Select a window of characters before and after the redaction,
            #which will provide a context of words to create a sentence vector from
            start=df2['start'][blank_iter]
            end=df2['end'][blank_iter]
            window=100
            if (start-window >=0):
                start_window=start-window
            else:
                start_window=0
            if (end+window<len(document.text)):
                end_window=end+window
            else:
                end_window=len(document.text)-1
            context=document.text[start_window:end_window]

            #Find the sentences containing the redaction.
            nlpcontext=nlp(context)
            sentences=list(nlpcontext.sents)
            sentences[0]=replace_block(str(sentences[0]))
            #Or, just use all the words in the window
            myconcat=''
            for i in sentences:
                myconcat = myconcat+' '+replace_block(str(i))

            #*****************************************************************
            #In this block, clean up the context in the same manner used
            #when making the model: remove punctuation, capitals, stopwords, etc.
            #This code block taken from 'Celebrity Word Vectors',
            #http://engineering.pmc.com/2017/08/celebrity-word-vectors-79/

            df=pd.Series(myconcat)
            raw_data = list(zip(df))
            df=pd.DataFrame(raw_data)
            df.columns=['text']
            # set up list for collecting cleaned content into
            results_all = []

            # pull the df into a list to loop through (seems quicker then looping through the pd df itself)
            raw_data = list(zip(df.text))

            # loop through each row of df and clean strip out the html tags
            for text in raw_data:

                # use try block to ignore errors in cleaning, should be fine as have lots of data
                    #result = remove_tags(text[0])
                results_all.append(text[0])
            # create a df with outputs
            df = pd.DataFrame(results_all,columns=['text'])
            # remove some messy artifacts
            df.text = df.text.str.replace("Bio:\n"," ")
            df.text = df.text.str.replace('\]\[\"',"")
            df.text = df.text.str.replace("&#8217;s"," ")
            df.text = df.text.str.replace("&#8216;s"," ")
            df.text = df.text.str.replace("&#8217;"," ")
            df.text = df.text.str.replace("&#8216;"," ")
            df.text = df.text.str.replace("&#8220;"," ")
            df.text = df.text.str.replace("&#8221;"," ")
            df.text = df.text.str.replace("&#8212;"," ")
            df.text = df.text.str.replace("&#039;"," ")
            df.text = df.text.str.replace("&#8211;"," ")
            df.text = df.text.str.replace("&#8230;"," ")

            # do some string cleaning directly on the df
            df.text = df.text.str.lower() # set all to lower
            df.text = df.text.str.replace("'s "," ") # remove 's from end of words
            df.text = df.text.str.replace("'s,",",") # remove 's from end of words
            df.text = df.text.str.replace(' \d+ ', ' ') # replace all words that are numbers - they wont be useful for us
            df.text = df.text.str.replace("' "," ") # remove '
            df.text = df.text.str.replace(" '"," ")
            df.text = df.text.str.replace(", "," , ")
            df.text = df.text.str.replace(",","")
            df.text = df.text.str.replace("!",".")
            df.text = df.text.str.replace("’s "," ")
            df.text = df.text.str.replace("’s,",",")
            df.text = df.text.str.replace("’s","")
            df.text = df.text.str.replace('"','')
            df.text = df.text.str.replace(' -- ',' ')
            df.text = df.text.str.replace(' ‘',' ')
            df.text = df.text.str.replace('’ ',' ')
            df.text = df.text.str.replace(': ',' ')
            df.text = df.text.str.replace('—',' ')
            df.text = df.text.str.replace(r'(\[.*\])', ' ') # remove anything inside [] as is usually leftover html junk
            df.text = df.text.str.replace('\n\n','\n')
            df.text = df.text.str.replace('\n',' ') # remove newlines within each article as will introduce dirty data later if left in
            df.text = df.text.str.replace('❤','love')
            df.text = df.text.str.replace('(','( ')
            df.text = df.text.str.replace(')',' )')
            df.text = df.text.str.replace('\. ',' . ')
            df.text = df.text.str.replace('“',' ')
            df.text = df.text.str.replace('”',' ')
            df.text = df.text.str.replace('\xa0',' ')
            df.text = df.text.str.replace('  ',' ')
            df.text = df.text.str.replace(r'(https://www.instagram.com.*? )', 'instagram ')
            df.text = df.text.str.replace(r'(https://instagram.com.*/? )', 'instagram ')
            df.text = df.text.str.replace(r'(https://www.twitter.com.*? )', 'twitter ')
            df.text = df.text.str.replace(r'(https://twitter.com.*? )', 'twitter ')
            df.text = df.text.str.replace(r'(https://www.youtube.com.*? )', 'youtube ')
            df.text = df.text.str.replace('?',' ?')
            df.text = df.text.str.replace('\\\\n',' ')
            df.text = df.text.str.replace('&amp;','and')
            df.text = df.text.str.replace('\\\ ',' ')
            df.text = df.text.str.replace('’ ',' ’ ')
            df.text = df.text.str.replace(' ‘',' ‘ ')
            df.text = df.text.str.replace(' pic ',' ')
            df.text = df.text.str.replace(' pics ',' ')

            # replace any double white spaces we might be left with
            df.text = df.text.str.replace('  ',' ')
            # get list of documents
            documents = list(df['text'])
            # get a list of single sentences
            #sentences = [s.split('.',1)[0].strip() for s in documents]
            sentences = documents # uncomment this to just use the whole dosument as one big sentence

            # create sentence stream iterator, removing stopwords and punctuation, also remove small sentences
            sentence_stream = [[str(i).translate(str.maketrans('','',string.punctuation)) for i in sentence.split(" ") if i not in stopwords.words('english')] for sentence in sentences]
            sentence_stream=sentence_stream[0]
            #******************************************************************************

            #strip the sentence stream/list of words down to only those present in the model
            sentence_stream2 = [i for i in sentence_stream if i in model.wv.vocab]

            #Begin output
            print("For redacted name number: ", blank_iter)
            print("In the context:")
            mysentences=list(nlpcontext.sents)
            print(mysentences)
            print("The replacement candidates are:")

            #The key of the analysis:
            #Find the similarity between a vector of words surrounding the redaction,
            #and a list of candidate celebrities whose bigram names are in the model
            rank = [model.n_similarity(sentence_stream2,[i]) for i in candidate_bigrams if i in model.wv.vocab]
            finalbigram=[i for i in candidate_bigrams if i in model.wv.vocab]
            #Sometimes there are no celebrities who fit the redaction pattern--
            if not finalbigram: print("No candidate replacements found")
            #Otherwise, output the proposed names and their similarity to the context
            if finalbigram:
                finaloutput=pd.DataFrame(finalbigram)
                finaloutput.columns=['name']
                finaloutput['similarity']=rank
                print(finaloutput.sort_values('similarity',ascending =False).head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="The directory to be redacted.")
    args = parser.parse_args()
    if args.input:
        main(args)
