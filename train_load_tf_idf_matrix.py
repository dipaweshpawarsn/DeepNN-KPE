
# coding: utf-8

# In[4]:


import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# In[5]:


def train_tf_idf(path_to_corpus,serialization_path,stop_words):
    file_names=os.listdir(path_to_corpus)
    no_files_thrown_exception_tfidf=0
    corpus=[]
    
    for i in range(0,len(file_names)):
        if not file_names[i].endswith('.txt'):
            continue
        try:
            text=str(open(path_to_corpus+file_names[i],'rb').read()).replace('\n','. ').replace('..','.')
        except:
            no_files_thrown_exception_tfidf+=1
            continue
        corpus.append(text)
    print('No Of files read: '+str(len(corpus)))
    print('No of files thrown exception while training tfIdf: '+str(no_files_thrown_exception_tfidf))

    # initialize count vectorizer
    cv=CountVectorizer(ngram_range=(1,4),lowercase=True,stop_words=stop_words,decode_error='ignore')
    cv_matrix=cv.fit_transform(corpus) # vocabulary by documents matrix # each cell denotes no of times that word appear in that perticular document
    print('Count Vectorizer Matrix done')
    
    #initialize tf-idf transformer
    idf=TfidfTransformer(smooth_idf=True,use_idf=True)
    idf.fit(cv_matrix)
    print('idf  done')

    #serialize objects to file
    output_file=open(serialization_path,'wb')
    pickle.dump(cv,output_file)
    pickle.dump(idf ,output_file)
    print('serialization Done')
    output_file.close()
    
    return (cv,idf)


# In[3]:


def load_cv_idf(serialization_path):
    
    f=open(serialization_path,'rb')
    cv=pickle.load(f)
    idf=pickle.load(f)
    f.close()
    return (cv,idf)

