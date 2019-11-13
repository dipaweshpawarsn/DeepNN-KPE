
# coding: utf-8

# In[ ]:


from train_load_tf_idf_matrix import *
import spacy
import pandas as pd
import math
import sys

def compute_avg_similarity(ckp_doc,ckps_docs):
    
    #candidate key phrase
    ckp_vec=ckp_doc.vector_norm
    avg_similarity=float(0)
    no_ckps_considered_for_similarity=0
    
    #current key phrase  
    for current_doc in ckps_docs:

        if ckp_doc.text.lower()!=current_doc.text.lower():
            if ckp_doc.vector_norm and current_doc.vector_norm:
                similarity=ckp_doc.similarity(current_doc)
                #print('nc1: '+current_chunk.text+' nc2: '+chunk.text+' similarity: '+str(similarity))
            else:
                similarity=0
            avg_similarity+=similarity
                #print('avg: '+str(avg_similarity))
            no_ckps_considered_for_similarity+=1
    
    if no_ckps_considered_for_similarity==0:
        return float(0)
    else:
        return avg_similarity/no_ckps_considered_for_similarity

def compute_features_lables(text,candidate_keyphrases,keyphrases_count_map,cv_idf,total_kp_count_in_training,actual_kps,nlp):
    
    #list of chunk feature vector(list) So its a list of lists
    feature_vector_phrases=[]
    labels=[]
    actual_candidates=[]

    #Map of chunk to its position in feature vector phrases list
    ckp_position_map={}

    #index
    feature_vector_phrases_index=0
    
    #extract count vectorizer and tf-idf transformer objects
    cv=cv_idf[0]
    idf=cv_idf[1]
    
    # transform current document
    count_vector=cv.transform([text])
    tf_idf_vector=idf.transform(count_vector)
    
    #store tf-idf scores in dataframe
    tfidf_df=pd.DataFrame(tf_idf_vector[0].T.todense(),index=cv.get_feature_names(),columns=["tfidf"])
    
    #compute spread(distance between 1st and last occurance) and keyphraseness
    #To compute spread use dictionary of nc to position where its feature vector is placed
    no_of_chunk_already_present=0

    no_ckps=len(candidate_keyphrases)
    ckps_docs=[]
    for candidate_kp_tuple in candidate_keyphrases:
        candidate_kp=candidate_kp_tuple[0]
        candidate_kp_doc=nlp(candidate_kp)
        ckps_docs.append(candidate_kp_doc)
    print('All documents formation done')
    
    for i, ckp_tuple in enumerate(candidate_keyphrases):
        
        feature_vector_phrase=[]
        ckp=ckp_tuple[0]
        ckp_prob=ckp_tuple[1]
        ckp_lowercase=ckp.lower()
        ckp_current_occurance_position=i
        
        #update spread of the chunk if it is already present
        if ckp_lowercase in ckp_position_map:
            no_of_chunk_already_present+=1
            ckp_first_position_and_index=ckp_position_map[ckp_lowercase].split('_')
            ckp_first_position=float(ckp_first_position_and_index[0])
            ckp_index=int(ckp_first_position_and_index[1]) # chunk index in feature vector phrases
            
            feature_vector_phrases[ckp_index][303]=(ckp_current_occurance_position - ckp_first_position)/no_ckps
            
            continue
        else:
            #this chunk is occuring 1st time
            ckp_position_map[ckp_lowercase]=str(ckp_current_occurance_position)+'_'+str(feature_vector_phrases_index)
            
        
        #phrase vector
        ckp_doc=ckps_docs[i]
        feature_vector_phrase.extend(list(ckp_doc.vector))
        
        #phrase length(301th)
        feature_vector_phrase.append(len(ckp.split()))
        
        #Normalized Phrase position(302)
        feature_vector_phrase.append(ckp_current_occurance_position/no_ckps)
        
         #compute TF-IDF (303th)
        if ckp_lowercase in tfidf_df.index:
            phrase_tfidf=tfidf_df.loc[ckp_lowercase,'tfidf']
        else:
            phrase_tfidf=0
        feature_vector_phrase.append(phrase_tfidf) 
        
        #compute spread, intially it would be chunk's.... if phrase occuras again then spread will be updated
        feature_vector_phrase.append(float(0))
        
        #compute keyphraseness (305)
        ckp_frequency_in_training=0.0
        if ckp_lowercase in keyphrases_count_map:
            ckp_frequency_in_training=keyphrases_count_map[ckp_lowercase]
        keyphraseness=ckp_frequency_in_training/(total_kp_count_in_training)
        feature_vector_phrase.append(keyphraseness)
        
        #compute candidate's avg similarity from other candidates ( If a candidate is key phrase, it should have less avg distance as key phrase is assumed to represent an article) (306)
        avg_similarity=compute_avg_similarity(ckp_doc, ckps_docs)
        feature_vector_phrase.append(avg_similarity)                           
        
        feature_vector_phrase.append(ckp_prob)

        feature_vector_phrases.append(feature_vector_phrase)
        feature_vector_phrases_index+=1
        
        if actual_kps is not None:
            if ckp_lowercase.strip() in actual_kps:
                labels.append(1)
            else:
                labels.append(0)
        
        actual_candidates.append(ckp_lowercase)
    #print('no of chunk laready present and hence not processed for feature computrations: '+str(no_of_chunk_already_present))   
    return (feature_vector_phrases,labels,actual_candidates) 