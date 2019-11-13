
# coding: utf-8

# In[ ]:

train_data_path='/mnt/lucydrive/KeyPhraseExtraction/Rev-oldData-V1/train/'

test_data_path='/mnt/lucydrive/KeyPhraseExtraction/Rev-oldData-V1/test/'

keyphrases_output_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/Experiment7/output/test/'

#tf-idf values will be saved to following path
serialization_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/TFIDF2.pickle'

file_for_training_deep_model='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/training-file.csv'

deep_model_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/Experiment7/data-model.h5'

keyphraseness_serialization_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/keyphraseness.pickle'

pickle_files_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/'

no_of_keyPhrases_in_output=20
no_of_epochs=1000
batch_size=5000

#test
actualKW_files_path='/mnt/lucydrive/KeyPhraseExtraction/Rev-oldData-V1/test/'
systemKW_files_path='/mnt/lucydrive/KeyPhraseExtraction/SNTPSRevAnalyzer/DataV1/FeatureV2/unsupervised-candidateGeneration/Experiment7/output/test/'