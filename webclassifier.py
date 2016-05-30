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
import sklearn.metrics
import numpy as np
from argparse import ArgumentParser

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def __setup_argument_parser():          
    #parses the command line arguments to get the training and testing file
    parser = ArgumentParser()
    parser.add_argument("-tr", "--train",
        help="Specify file for training")
    parser.add_argument("-te", "--test",
        help="Specify file for testing")
    parser.add_argument("-o", "--out",
        help="Specify file for writing the predictions")
    parser.add_argument("-n", "--ngram",type = int,
        help="Specify the max number for n-grams : 1 for unigram, 2 for bigram etc, default = 1", default = 1)
    return parser.parse_args()

def __read_training_file(filename):
  # reads the training file and outputs the set of keywords and the corresponding label for each url
  country = "data/data_country_codes.csv"
  df = pd.read_csv(country)

  country_list = ['US','GB','IN','AU']
  labels = []
  keywords = []

  f = codecs.open(filename,'r',encoding='ISO-8859-1')
  for line in f:
    arr = line.split("||")
    if len(arr) >= 6:
        if is_number(arr[1]) and int(arr[1])<400:
            if len(df.loc[df['FULL_URL'] == arr[0]]) == 0:
                country = 'US'
            else:
                country = df.loc[df['FULL_URL'] == arr[0]]['brm_country_code'].iloc[0]
            if country in country_list:
              keywords.append(arr[3]+" "+arr[4]+" "+arr[5])
              labels.append(arr[2])
  return (keywords,labels) 


def __tokenize_test(keywords):
  #performs preprocessing for the testing keywords and returns the tokens
  stop = stopwords.words('english')
  stop.append("na")
  stop.append("")
  tokens = []
  i = 0
  for word in keywords:
    token = textblob.TextBlob(word).words
    tokens.append(token)
  final_tokens = []
  final_labels = []
  for token in tokens:
    final_token = ''
    for word in token:
      word =  word.lemmatize()
      word =  word.lower()
      if word not in stop:
        final_token = final_token + word + " "
    if final_token != " " and final_token != "":
      final_tokens.append(final_token)
  return final_tokens

  

def __tokenize(keywords,labels):
  #performs preprocessing for the training keywords and returns the final tokens and final labels
  stop = stopwords.words('english')
  stop.append("na")
  stop.append("")
  tokens = []
  i = 0
  for word in keywords:
    token = textblob.TextBlob(word).words
    tokens.append(token)
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
  return (final_tokens,final_labels)

def __create_ngram_matrix(tokens,test_tokens,n):
  #returns the ngram count matrix for the training tokens and the test tokens
  vectorizer = CountVectorizer(input='content',ngram_range=(1,n),decode_error ='ignore', stop_words='english',analyzer='word')
  X = vectorizer.fit_transform(tokens+test_tokens)
  X = X.toarray()
  return X

def __train(X, labels):
  #trains the models on the training count matrix
  models = [LogisticRegression(C=1,solver = 'lbfgs', multi_class = 'multinomial'), MultinomialNB(),RandomForestClassifier(n_estimators = 300)]
  for model in models:
    model.fit(X,labels)
  return models

def __read_testing_file(filename):
  #reads the testing file and outputs the set of keywords for each url
  country = "data/data_country_codes.csv"
  df = pd.read_csv(country)
  country_list = ['US','GB','IN','AU']
  labels = []
  keywords = []
  urls = []
  f = codecs.open(filename,'r',encoding='ISO-8859-1')
  for line in f:
    arr = line.split("||")
    if len(arr) >= 5:
        if is_number(arr[1]) and int(arr[1])<400:
            if len(df.loc[df['FULL_URL'] == arr[0]]) == 0:
                country = 'US'
            else:
                country = df.loc[df['FULL_URL'] == arr[0]]['brm_country_code'].iloc[0]
            if country in country_list:
              urls.append(arr[0])
              keywords.append(arr[2]+" "+arr[3]+" "+arr[4])

  return urls,keywords

def predict(test,models):
  # returns the predictions for the given test matrix
  preds = np.zeros((len(models),len(test)))
  prob = np.zeros((len(models),len(test)))
  for i in range(len(models)):
    preds[i]=models[i].predict(test)
    prob[i] = models[i].predict_proba(test).max(1)


  final_preds = np.zeros(len(test))
  final_prob = np.zeros(len(test))
  for i in range(len(test)):
     temp = Counter(preds[:,i]).most_common(1)[0]
     if temp[1] == 1:
        final_preds[i] = preds[0,i]
        final_prob[i] = prob[0,i]
     else:
        final_preds[i] = temp[0]
        final_prob[i] = prob[np.argmax(preds[:,i]),i]
  return (final_preds,final_prob)

def __write_predictions(urls,predictions, probabilities,out_file):
  # writes the predictions to the output file
  with open(out_file,'w') as f:
    for (url,pred,prob) in zip(urls,predictions,probabilities):
      f.write('%s,%f,%f\n'%(url,pred,prob))
  
def main():
  parsed_args = __setup_argument_parser()
  training_file = parsed_args.train
  testing_file = parsed_args.test
  out_file = parsed_args.out
  n = parsed_args.ngram
  keywords, labels = __read_training_file(training_file)
  final_tokens, labels = __tokenize(keywords,labels)
  urls,test = __read_testing_file(testing_file)
  test_tokens = __tokenize_test(test)
  matrix = __create_ngram_matrix(final_tokens,test_tokens,n)
  models = __train(matrix[:len(final_tokens)],labels)
  predictions, probabilities = predict(matrix[len(final_tokens):],models)
  __write_predictions(urls,predictions,probabilities,out_file)

if __name__ == '__main__':
  main()
  
