import string
import re
import pke
import string
from nltk.corpus import stopwords
import os
import traceback


                

def generate_candidate(path_to_doc):
    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=path_to_doc,language='en', normalization=None,encoding = 'unicode_escape')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_weighting(alpha=1.1,threshold=0.74,method='average')
    keyphrases = extractor.get_n_best(1000)

    return keyphrases