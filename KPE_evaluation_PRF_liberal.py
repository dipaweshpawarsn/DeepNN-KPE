#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
from config import *

#actualKW_files_path='D:/RevAnalyzer/TestingSet/'
#systemKW_files_path='D:/RevAnalyzer/oldDataResults/PositionRank/'
actualKW_files=[f for f in os.listdir(actualKW_files_path) if f.endswith('.key')]
systemKW_files=[f for f in os.listdir(systemKW_files_path) if f.endswith('.key')]

actualKW_files.sort()
systemKW_files.sort()


# In[72]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()

avg_precision=0.0
avg_recall=0.0
avg_fmeasure=0.0
no_of_files=len(actualKW_files)

for i in range(no_of_files):
    
    system_kws=[kw.lower().strip() for kw in open(systemKW_files_path+systemKW_files[i],encoding='unicode_escape').read().split('\n')]
    actual_kws=[kw.lower().strip() for kw in open(actualKW_files_path+actualKW_files[i],encoding='unicode_escape').read().split('\n')]
    system_words=set([ps.stem(w) for kw in system_kws for w in kw.split()])
    actual_words=set([ps.stem(w) for kw in actual_kws for w in kw.split()])
#     print('actual file: '+actualKW_files[i])
#     print('system file: '+systemKW_files[i])
    no_of_actual_words=len(actual_words)
    no_of_system_words=len(system_words)
    no_of_correctly_identified_words=0

    for w in system_words:
        if w in actual_words and w.strip()!='':
            no_of_correctly_identified_words+=1
    
    if no_of_system_words !=0:
        precision=no_of_correctly_identified_words/(no_of_system_words)
    else:
        precision=0
    if no_of_actual_words!=0:
        recall=no_of_correctly_identified_words/(no_of_actual_words)
    else:
        recall=0
        
    if precision==0 and recall==0:
        fmeasure=0
    else:
        fmeasure=2*precision*recall/(precision+recall)
    
    avg_precision+=precision
    avg_recall+=recall
    #print('precision: '+str(precision))
#     print('recall: '+str(recall))
#     print('fmeasure: '+str(fmeasure))
#     print('No Of actual KWs: '+str(no_of_actual_kws))
#     print('No Of system KWs: '+str(no_of_system_kws))
#     print('No Of match kws: '+str(no_of_correctly_identified_kws))

avg_precision=avg_precision/no_of_files
avg_recall=avg_recall/no_of_files
if avg_precision==0 and avg_recall==0:
    avg_fmeasure=0.0
else:
    avg_fmeasure=2*avg_precision*avg_recall/(avg_precision+avg_recall)


# In[73]:


print('precision: '+str(avg_precision))
print('recall: '+str(avg_recall))
print('fmeasure: '+str(avg_fmeasure))


# In[ ]:




