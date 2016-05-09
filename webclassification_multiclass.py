from sklearn.feature_extraction.text import CountVectorizer
import os
import random
from collections import defaultdict
from collections import Counter
import itertools
import numpy as np
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sets import Set
import textblob
import codecs
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import sklearn.metrics
import numpy as np


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

files = ["webclassification.csv", "UT_data.csv"]

labels = []
keywords = []

for filename in files:
  f = codecs.open(filename,'r',encoding='ISO-8859-1')
  for line in f:
    arr = line.split("||")
    if len(arr) >= 6:
        if is_number(arr[1]) and int(arr[1])<400:
            keywords.append(arr[3]+" "+arr[4]+" "+arr[5])
            labels.append(arr[2])

stop = stopwords.words('english')
stop.append("na")
stop.append("")

tokens = []
i = 0
for word in keywords:
  token = textblob.TextBlob(word).words
  tokens.append(token)
  i+=1



final_tokens = []
final_labels = []
for (token,label) in zip(tokens,labels):
  final_token = ''
  for word in token:
    word =  word.lemmatize()
    word =  word.lower()
    if word not in stop:
      final_token = final_token + word + " "
  if final_token != " " and final_token != "":
    final_tokens.append(final_token)
    final_labels.append(label)
labels = final_labels
vectorizer = CountVectorizer(input='content',ngram_range=(1,2),decode_error ='ignore', stop_words='english',analyzer='word')
X = vectorizer.fit_transform(final_tokens)
X = X.toarray()

models = [RandomForestClassifier(n_estimators = 300)]
#models = [LogisticRegression(C=1,solver = 'lbfgs', multi_class = 'multinomial'), MultinomialNB(),RandomForestClassifier(n_estimators = 300)]
folds = 5
n = len(models)
kfolder=StratifiedKFold(labels, n_folds= folds,shuffle=True)     
j=0
mean_accuracy = np.zeros(n)
mean_f1_macro = np.zeros(n)
mean_f1_micro = np.zeros(n)
mean_f1_weighted = np.zeros(n) 
ensemble_accuracy = 0
ensemble_f1_macro = 0
ensemble_f1_micro = 0
ensemble_f1_weighted = 0
label_set = [1,2,3,4,5,6,7,11,10003,10008]
for train_index, test_index in kfolder: 

    X_train, X_cv = X[train_index], X[test_index]
    y_train, y_cv = np.array(labels, dtype = int)[train_index], np.array(labels,dtype=int)[test_index]
    preds = np.zeros((len(models),len(y_cv)))
    print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
    
    for i in range(len(models)):
      print "training"
      models[i].fit(X_train,y_train) 
      print "testing" 
      preds[i]=models[i].predict(X_cv)
      probabilities = models[i].predict_proba(X_cv)
      #print sorted(probabilities[:,0])
      for index in range(10):
        for k in range(1,11):
          temp = (k * len(y_cv)) / (100)
          sorted_labels = [x for y,x in sorted(zip(probabilities[:,index],y_cv),reverse=True)]
          precision =  float(sorted_labels[0:temp].count(label_set[index]))/float(temp)
          #print label_set[index],precision
      #print models[i].classes_
      mean_accuracy[i] += sklearn.metrics.accuracy_score(y_cv, preds[i])
      mean_f1_macro[i] += sklearn.metrics.f1_score(y_cv, preds[i], average = 'macro')
      mean_f1_micro[i] += sklearn.metrics.f1_score(y_cv, preds[i], average = 'micro')
      mean_f1_weighted[i] += sklearn.metrics.f1_score(y_cv, preds[i], average = 'weighted')

    final_preds = np.zeros(len(y_cv))
    for i in range(len(y_cv)):
       temp = Counter(preds[:,i]).most_common(1)[0]
       if temp[1] == 1:
          final_preds[i] = preds[0,i]
       else:
          final_preds[i] = temp[0] 
    
    for i in range(len(final_preds)):
        if final_preds[i] != y_cv[i]:
            try:
              print final_tokens[test_index[i]], y_cv[i], final_preds[i]
            except: 
              print "sorry"
    #print sklearn.metrics.confusion_matrix(y_cv, final_preds, labels = [1,2,3,4,5,6,7,11,10003,10008])
    ensemble_accuracy += sklearn.metrics.accuracy_score(y_cv, final_preds)
    ensemble_f1_macro += sklearn.metrics.f1_score(y_cv, final_preds, average = 'macro')
    ensemble_f1_micro += sklearn.metrics.f1_score(y_cv, final_preds, average = 'micro')
    ensemble_f1_weighted += sklearn.metrics.f1_score(y_cv, final_preds, average = 'weighted')
    j+=1

    
mean_accuracy/=j
mean_f1_macro /= j
mean_f1_micro /= j
mean_f1_weighted /= j
ensemble_accuracy /= j
ensemble_f1_macro /= j
ensemble_f1_micro /= j
ensemble_f1_weighted /= j

print mean_accuracy
print mean_f1_macro
print mean_f1_micro
print mean_f1_weighted 
print ensemble_accuracy
print ensemble_f1_macro
print ensemble_f1_micro
print ensemble_f1_weighted
