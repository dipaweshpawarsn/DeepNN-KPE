
# coding: utf-8

# In[30]:


import os
import spacy
from feature_engineering import *
from candidate_generation import *
from train_load_tf_idf_matrix import *
from sklearn.feature_extraction.text import TfidfTransformer
from IPython.display import clear_output
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import re
import string
from config import *
import pickle


# In[28]:


def load_keyPhrases_count_map(train_path,file_names):
    
    keyphrases_count_map={}
    total_kp_count=0
    for i in range(0,len(file_names)):
        
        if not file_names[i].endswith('.key'):
            continue
        
        key_phrases=open(train_path+file_names[i],'r',encoding='unicode_escape').read().lower().split('\n')
         # each key_phrases array consists of one blank key phrase
        
        for kp in key_phrases:
            
            if kp.strip()=='':
                continue
            total_kp_count+=1
            if kp in keyphrases_count_map:
                keyphrases_count_map[kp]+=1
            else:
                keyphrases_count_map[kp]=1
    return (keyphrases_count_map,total_kp_count)


# In[3]:


#train_path='/mnt/lucydrive/KeyPhraseExtraction/MAUI/TrainingSet/'
#serialization_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/TFIDF.pickle'
#train_file_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/Experiment18/revised-training-file_remaining.csv'
#file_for_training_deep_model='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/Experiment2/training-file_remaining.csv'
train_file=open(file_for_training_deep_model,'w')
train_file_names=os.listdir(train_data_path)
train_file_names.sort()

#no_of_max_features=2500000
stop_words=['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
            'yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they',
            'them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are',
            'was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and',
            'but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into',
            'through','during','before','after','above','below','to','from','up','down','in','out','on','off','over',
            'under','again','further','then','once','here','there','when','where','why','how','all','any','both','each',
            'few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s',
            't','can','will','just','don','should','now','a','able','about','across','after','all','almost','also','am',
            'among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did',
            'do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him',
            'his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me',
            'might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own',
            'rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then',
            'there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where',
            'which','while','who','whom','why','will','with','would','yet','you','your']

#load spacy model
nlp=spacy.load('en_core_web_lg')
nlp.remove_pipe('ner')

if 'TFIDF2.pickle' not in os.listdir(pickle_files_path):
    #train count vectorizer and tf-idf vectorzier
    cv_idf=train_tf_idf(train_data_path,serialization_path,stop_words)
    #cv_idf=load_cv_idf(serialization_path)
else:
    cv_idf=load_cv_idf(serialization_path)
print('trained tf-idf')


if 'keyphraseness.pickle' not in os.listdir(pickle_files_path):
    keyphrases_count_map_and_total_kps=load_keyPhrases_count_map(train_data_path,train_file_names)
    keyphraseness_file=open(keyphraseness_serialization_path,'wb')
    keyphrases_count_map=keyphrases_count_map_and_total_kps[0]
    total_kp_count_in_training=keyphrases_count_map_and_total_kps[1]
    pickle.dump(keyphrases_count_map,keyphraseness_file)
    pickle.dump(total_kp_count_in_training,keyphraseness_file)
    keyphraseness_file.close()
else:
    keyphraseness_file=open(keyphraseness_serialization_path,'rb')
    keyphrases_count_map=pickle.load(keyphraseness_file)
    total_kp_count_in_training=pickle.load(keyphraseness_file)
    keyphraseness_file.close()
#print('total_kp_count_in_training'+str(total_kp_count_in_training))
#print(keyphrases_count_map)
print('counted keyphrases')

for i in range(1210,len(train_file_names),2):
    
    clear_output(wait=True)
    print('Text File going on: '+train_file_names[i+1])
    print('Key File going on: '+train_file_names[i])
    print(str(i/2)+' th file is going on')
    
    try:
        text_file=open(train_data_path+train_file_names[i+1],'r',encoding='unicode_escape')
        key_file=open(train_data_path+train_file_names[i],'r',encoding='unicode_escape')
        text=text_file.read().replace('\n','. ').replace('..','.')
        keywords=set([k.lower().strip() for k in key_file.read().split('\n') if k!='' and k!=None])
        text_file.close()
        key_file.close()
        unsupervised_keyphrases=generate_candidate(text)
        print('candidate Generated')
    except Exception as e:
        print('file lead to exception'+str(e))
        continue
    feature_vector_phrases_and_labels=compute_features_lables(text,unsupervised_keyphrases,keyphrases_count_map,cv_idf,total_kp_count_in_training,keywords,nlp)
    print('Feature computed')
    feature_vector_phrases=feature_vector_phrases_and_labels[0]
    lables=feature_vector_phrases_and_labels[1]
    actual_candidates=feature_vector_phrases_and_labels[2]
    sep='@|@|@'
    for j,fv in enumerate(feature_vector_phrases):
        if len(fv)==0:
            continue
        features=''
        for f in fv:
            features+=str(f)+sep
        train_file.write(actual_candidates[j]+sep+features+str(lables[j])+'\n')
    train_file.flush()
train_file.close()