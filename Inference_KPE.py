
from keras.models import load_model
import spacy
from config import *
import pickle
import os
import string
import re
from feature_engineering import *
from candidate_generation import *
from train_load_tf_idf_matrix import *
from sklearn.feature_extraction.text import TfidfTransformer
import numpy

model = load_model(deep_model_path)
#train_files=os.listdir(train_data_path)

nlp=spacy.load('en_core_web_lg')
nlp.remove_pipe('ner')

cv_idf=load_cv_idf(serialization_path)

keyphraseness_file=open(keyphraseness_serialization_path,'rb')
keyphrase_count_map=pickle.load(keyphraseness_file)
total_kp_count_in_training=pickle.load(keyphraseness_file)
keyphraseness_file.close()

print('test Data path '+test_data_path)
print('output path '+keyphrases_output_path)

test_files=os.listdir(test_data_path)
no_of_files_processed=0



for f in test_files:
    if f.endswith('.txt'):
        text_file=open(test_data_path+f,'r',encoding='unicode_escape')
        output_file=open(keyphrases_output_path+f.replace('.txt','.key'),'w')
        output_file_with_probabilities=open(keyphrases_output_path+f.replace('.txt','.key_withProbabilities'),'w')
        text=text_file.read().replace('\n','. ').replace('..','.')
        text_file.close()

        unsupervised_keyphrases=generate_candidate(text)
        feature_vector_phrases_and_labels=compute_features_lables(text,unsupervised_keyphrases,keyphrase_count_map,cv_idf,total_kp_count_in_training,None,nlp)
        feature_vector_phrases=feature_vector_phrases_and_labels[0]
        actual_candidates=feature_vector_phrases_and_labels[2]
        
        test_samples=numpy.array([numpy.array(fv) for fv in feature_vector_phrases])
        predictions=list(model.predict(test_samples[:,300:307])[:,0])
        keyphraseness_probabilities=dict(zip(actual_candidates,predictions))
        
        #sort according to probabilities
        keyphraseness_probabilities_sorted=sorted(keyphraseness_probabilities.items(),key=lambda kp:kp[1],reverse=True)
        
        for kp in keyphraseness_probabilities_sorted[0:20]:
            output_file.write(kp[0]+'\n')
            output_file_with_probabilities.write(kp[0]+'\t'+str(kp[1])+'\n')
        output_file.close()
        output_file_with_probabilities.close()
        no_of_files_processed+=1
        print(no_of_files_processed)