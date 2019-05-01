import pytest
import argparse
import glob
import nltk
import spacy
import numpy as np
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
import os
from operator import itemgetter

def test_names():
    nlp = spacy.load('en')
    document = nlp(u"My name is Brandon")
    toredact = [(e.text, e.start_char, e.end_char, e.label_) for e in filter(lambda w: w.label_ == 'PERSON', document.ents)]

    print(toredact)
    assert toredact[0][0] == 'Brandon'

