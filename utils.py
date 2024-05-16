## Run ML methods on PanPred and panta outputs 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import svm
import random
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
from collections import Counter
from itertools import groupby

import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError

def run_ML(X, y, data_set, approach="Default"):
    # X is numpy as
    X = np.array(X)
    base_dir = 'results'
    if not os.path.isdir(base_dir):
        os.system('mkdir '+ base_dir)
    score = []
    methods = []
    n_loops = 1
    n_folds = 5
    n_samples = y.shape[0]
    for i in range(n_loops):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state = i)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            path_dir = base_dir +'/' + data_set + '_run_'+str(i)+'_'+ 'fold_'+str(fold)+'_'+approach
            print('Run: ', i, ', fold: ', fold)
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            print("Train freq: ", [len(list(group)) for key, group in groupby(sorted(y_train))])
            
            # if i <= 0 and fold <= 0:
            #     print("n_samples: ", n_samples)
            #     print("Reduced shape of the data: ", X_train.shape, X_test.shape)
            # print(test_idx[:10])

            # SVM
            # methods.append('SVM')
            # print(methods[-1], end =', ')
            # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            # np.savetxt(path_dir + "_SVM_labels.csv", y_predict, delimiter=",")
            # score.append(f1_score(y_predict, y_test, average='macro'))

#             # Decision Tree
#             methods.append('Decision Tree')
#             print(methods[-1], end =', ')
#             clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_DecisionTree_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # RF
#             methods.append('RF')
#             print(methods[-1], end =', ')
#             clf = RandomForestClassifier().fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_RandomForest_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Neural network
#             methods.append('Neural network')
#             print(methods[-1], end =', ')
#             clf = MLPClassifier(alpha=1, max_iter=2000).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_NeuralNet_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Adaboost
#             methods.append('Adaboost')
#             print(methods[-1], end =', ')
#             clf = AdaBoostClassifier().fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_Adaboost_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

            ## K-NN 
            methods.append('kNN')
            print(methods[-1], end =', ')
            clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            np.savetxt(path_dir + "_NearestNeighbors_labels.csv", y_predict, delimiter=",")
            score.append(f1_score(y_predict, y_test, average='macro'))

            # # Naive Bayes
            # methods.append('NaiveBayes')
            # print(methods[-1], end ='\n')
            # clf = GaussianNB().fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            # np.savetxt(path_dir + "_NaiveBayes_labels.csv", y_predict, delimiter=",")
            # score.append(f1_score(y_predict, y_test, average='macro'))
   
            # Xgboost
#             clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=500, objective='binary:logistic', booster='gbtree', use_label_encoder=False) #binary
#             methods.append('Xgboost')
#             print(methods[-1], end =', ')
#             XGB=clf.fit(X_train,y_train)
#             y_predict=XGB.predict(X_test)
#             np.savetxt(path_dir + "_Xgboost_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))
            
            # GradientBoostingClassifier
            # methods.append('Gradient Boost Decision Tree')
            # print(methods[-1], end =', ')
            # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0).fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            # np.savetxt(path_dir + "_GBDT_labels.csv", y_predict, delimiter=",")
            # score.append(f1_score(y_predict, y_test, average='macro'))
                  
            methods.append('LightGBM')
            print(methods[-1], end =', ')
            model = lgb.LGBMClassifier(verbose=-1)
            model.fit(X_train, y_train)
            y_predict=model.predict(X_test) 
            np.savetxt(path_dir + "_LightGBM_labels.csv", y_predict, delimiter=",")
            score.append(f1_score(y_predict, y_test, average='macro'))
        
    # Print statistics
    n_methods = len(set(methods))
    score_np = np.array(score)
    # Each column is a method
    print(methods[:n_methods])
    average_score = np.mean(score_np.reshape((n_loops*n_folds, n_methods)), axis=0)
    print(np.round(average_score, 2))


### Feature extraction
import re
from urllib.parse import urlparse
import tldextract

def count_special_characters(url):
    special_chars = set(['-', '_', '.', '~', '!', '*', '\'', '(', ')', ';', ':', '@', '&', '=', '+', '$', ',', '/', '?', '#', '[', ']', '%'])
    count = sum(1 for char in url if char in special_chars)
    return count

def count_non_alphanumeric_characters(url):
    count = sum(1 for char in url if not char.isalnum())
    return count

def extract_top_level_domain(url):
    # Use tldextract to parse the URL
    extracted = tldextract.extract(url)
    return extracted.suffix    

def count_obfuscated_characters(url):
    # Regular expression pattern to match obfuscated characters
    obfuscated_pattern = r'%[0-9a-fA-F]{2}|\\x[0-9a-fA-F]{2}'

    # Find all matches of obfuscated patterns in the URL
    obfuscated_matches = re.findall(obfuscated_pattern, url)

    # Count the number of obfuscated characters
    num_obfuscated_characters = len(obfuscated_matches)

    return num_obfuscated_characters

def ratio_obfuscated_characters(url):
    # Regular expression pattern to match obfuscated characters
    obfuscated_pattern = r'%[0-9a-fA-F]{2}|\\x[0-9a-fA-F]{2}'

    # Find all matches of obfuscated patterns in the URL
    obfuscated_matches = re.findall(obfuscated_pattern, url)

    # Count the number of obfuscated characters
    num_obfuscated_characters = len(obfuscated_matches)

    return float(num_obfuscated_characters)/float(len(url))

def letter_ratio_in_url(url):
    # Count total characters and letters in the URL
    total_chars = len(url)
    letter_chars = sum(1 for char in url if char.isalpha())

    # Calculate letter ratio
    if total_chars > 0:
        letter_ratio = letter_chars / total_chars
    else:
        letter_ratio = 0.0  # Default to 0 if the URL is empty

    return letter_ratio

def digit_ratio_in_url(url):
    # Count total characters and digits in the URL
    total_chars = len(url)
    digit_chars = sum(1 for char in url if char.isdigit())

    # Calculate digit ratio
    if total_chars > 0:
        digit_ratio = digit_chars / total_chars
    else:
        digit_ratio = 0.0  # Default to 0 if the URL is empty

    return digit_ratio

def count_equals_in_url(url):
    # Count the number of '=' characters in the URL
    num_equals = url.count('=')
    return num_equals

def count_ampersand_in_url(url):
    # Count the number of '&' characters in the URL
    num_ampersand = url.count('&')
    return num_ampersand

# Not necessary
def char_continuation_rate(url):
    if len(url) == 0:
        return 0
    
    continuation_count = 0
    prev_char = url[0]
    
    # Count continuation of characters
    for char in url[1:]:
        if char == prev_char:
            continuation_count += 1
        prev_char = char
    
    # Calculate continuation rate
    continuation_rate = continuation_count / len(url)
    return continuation_rate

def url_char_prob(url):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_url = ''.join(char.lower() for char in url if char.isalnum())
    
    # Calculate character frequencies
    char_freq = Counter(cleaned_url)
    
    # Calculate total number of characters
    total_chars = len(cleaned_url)
    
    # Calculate character probabilities
    char_prob = {char: freq / total_chars for char, freq in char_freq.items()}
    
    return char_prob

def count_question_marks_in_url(url):
    # Count the number of '?' characters in the URL
    num_question_marks = url.count('?')
    return num_question_marks

def extract_features(url):
    # Parse the URL using urlparse
    parsed_url = urlparse(url)

    # Extract domain and path components
    domain = parsed_url.netloc
    path = parsed_url.path

    # Count number of subdomains
    subdomains = domain.split('.')
    num_subdomains = len(subdomains) - 1  # excluding the root domain

    # Check if the URL has an IP address as the domain (indicative of suspicious URLs)
    contains_ip = bool(re.match(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain))

    # Extract other features from path
    path_length = len(path)
    num_path_segments = len(path.strip('/').split('/'))

    # Check if URL uses HTTPS (indicative of secure connection)
    uses_https = 1 if parsed_url.scheme == 'https' else 0

    # Extract file extension (if applicable)
    file_extension = path.split('.')[-1] if '.' in path else ''

    # Construct feature dictionary
    features = {
        'domain': domain,
        'num_subdomains': num_subdomains,
        'contains_ip': int(contains_ip),
        'path_length': path_length,
        'num_path_segments': num_path_segments,
        'uses_https': uses_https,
        'file_extension': file_extension,
        'count_special_characters': count_special_characters(url),
        'count_non_alphanumeric_characters': count_non_alphanumeric_characters(url),
        'TLD': extract_top_level_domain(url),
        'count_obfuscated_characters': count_obfuscated_characters(url),
        'letter_ratio_in_url': letter_ratio_in_url(url),
        'digit_ratio_in_url': digit_ratio_in_url(url),
        'count_equals_in_url': count_equals_in_url(url),
        'NoOfAmpersandInURL': count_ampersand_in_url(url),
        'CharContinuationRate': char_continuation_rate(url),
        #'URLCharProb': url_char_prob(url),
        'ratio_obfuscated_characters': ratio_obfuscated_characters(url),
        'NoOfQMarkInURL':count_question_marks_in_url(url)
    }

    return features

def extract_numerical_features(url):
    # Parse the URL using urlparse
    parsed_url = urlparse(url)

    # Extract domain and path components
    domain = parsed_url.netloc
    path = parsed_url.path

    # Count number of subdomains
    subdomains = domain.split('.')
    num_subdomains = len(subdomains) - 1  # excluding the root domain

    # Check if the URL has an IP address as the domain (indicative of suspicious URLs)
    contains_ip = bool(re.match(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', domain))

    # Extract other features from path
    path_length = len(path)
    num_path_segments = len(path.strip('/').split('/'))

    # Check if URL uses HTTPS (indicative of secure connection)
    uses_https = 1 if parsed_url.scheme == 'https' else 0

    # Extract file extension (if applicable)
    file_extension = path.split('.')[-1] if '.' in path else ''

    # Construct feature dictionary
    features = {
        'num_subdomains': num_subdomains,
        'contains_ip': int(contains_ip),
        'path_length': path_length,
        'num_path_segments': num_path_segments,
        'uses_https': uses_https,
        'count_special_characters': count_special_characters(url),
        'count_non_alphanumeric_characters': count_non_alphanumeric_characters(url),
        'count_obfuscated_characters': count_obfuscated_characters(url),
        'letter_ratio_in_url': letter_ratio_in_url(url),
        'digit_ratio_in_url': digit_ratio_in_url(url),
        'count_equals_in_url': count_equals_in_url(url),
        'NoOfAmpersandInURL': count_ampersand_in_url(url),
        'CharContinuationRate': char_continuation_rate(url),
        #'URLCharProb': url_char_prob(url),
        'ratio_obfuscated_characters': ratio_obfuscated_characters(url),
        'NoOfQMarkInURL':count_question_marks_in_url(url)
    }

    return features

####### GNNPhish
import torch
from torch_geometric.data import Data, Dataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

# Nhan xet: Add arcs in both direction is worse than add arcs from root node to child nodes

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.linconcat = Linear(2*hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        y1 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        y1 = F.dropout(y1, p=0.5, training=self.training)
        x = torch.cat((x1, y1), dim=1)
        x = self.linconcat(x)
        
        return x

# URLNet prepare input
# newdf = pd.DataFrame({'label': labels, 'url': smalldata['url']})
# newdf.iloc[train_idx,:].to_csv('../URLNet/data/dataset_train.csv', header=None, index=None)
# newdf.iloc[test_idx,:].to_csv('../URLNet/data/dataset_test.csv', header=None, index=None)
# URLNet results
# results = pd.read_csv('../URLNet/runs/1000_emb3_dlm0_32dim_minwf1_1conv3456_5ep/train_1000_test_200.txt',sep='\t')
# print(f1_score(results.iloc[:,0], results.iloc[:,1], average='macro'))