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
from sklearn.linear_model import LogisticRegression
import random
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.model_selection import StratifiedKFold

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
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = i)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X,y)):
            path_dir = base_dir +'/' + data_set + '_run_'+str(i)+'_'+ 'fold_'+str(fold)+'_'+approach
            print('Run: ', i, ', fold: ', fold)
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            print("Train freq: ", [len(list(group)) for key, group in groupby(sorted(y_train))])
            
         
                  
            methods.append('LightGBM')
            print(methods[-1], end =', ')
            model = lgb.LGBMClassifier(verbose=-1)
            model.fit(X_train, y_train)
            y_predict=model.predict(X_test) 
            y_score = model.predict_proba(X_test)[:, 1]
            df = pd.DataFrame({
              'true_label': y_test,
              'predicted_label': y_predict,
              'y_score': y_score
            })
            df.to_csv(path_dir + "_LightGBM_labels.csv", index=False)
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

#import whois
from datetime import datetime

def get_domain_age(domain):
    try:
        # Lấy thông tin whois của domain
        domain_info = whois.whois(domain)
        
        # Lấy ngày tạo domain
        creation_date = domain_info.creation_date
        
        # Kiểm tra xem creation_date có giá trị không
        if len(creation_date) is not None:
            # Nếu ngày tạo là một danh sách, lấy phần tử đầu tiên
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            # Tính toán tuổi domain
            current_date = datetime.now()
            age = current_date - creation_date
            
            # Chuyển đổi tuổi thành năm
            age_years = age.days / 365.25
            
            return age_years
        else:
            return 0  # Trả về 0 nếu không có thông tin ngày tạo domain
    except Exception as e:
        print(f"Error: {e}")
        return 0  # Trả về 0 nếu có lỗi xảy ra trong quá trình lấy thông tin whois

def has_whois(domain):
    try:
        # Lấy thông tin whois của domain
        domain_info = whois.whois(domain)
        
        # Kiểm tra xem thông tin whois có tồn tại và có ngày tạo hay không
        if domain_info and domain_info.creation_date:
            return 1
        else:
            return 0
    except Exception as e:
        # Bắt các ngoại lệ (nếu domain không hợp lệ hoặc không tìm thấy thông tin whois)
        return 0


def detect_mouseover_phishing(url):
    try:
        # Gửi yêu cầu GET để lấy nội dung trang web
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm tất cả thẻ <a> có sự kiện onmouseover
        links_with_mouseover = soup.find_all('a', onmouseover=True)
        
        for link in links_with_mouseover:
            mouseover_event = link['onmouseover']
            # Kiểm tra nếu sự kiện onmouseover thay đổi href hoặc có dấu hiệu đáng ngờ
            if 'href' in mouseover_event:
                return 1
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0
from bs4 import BeautifulSoup
import requests

def get_form_count(url):
    try:
        # Gửi yêu cầu GET để lấy nội dung trang web
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        # Phân tích nội dung HTML của trang web
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm tất cả thẻ <form> trên trang web
        forms = soup.find_all('form')
        
        # Đếm số lượng form
        form_count = len(forms)
        
        return form_count
    except Exception as e:
        print(f"Error: {e}")
        return 0  # Trả về 0 nếu có lỗi xảy ra



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
        'NoOfQMarkInURL':count_question_marks_in_url(url),
        #'DomainLength': int(len(domain)),
        #'DomainAge': get_domain_age(domain),
        #'HasWhois': has_whois(domain),
       # 'UseMouseOver': detect_mouseover_phishing(url),
       # 'FormCount': get_form_count(url)
        
    }
    

    return features

def extract_numerical_features(url):
    # Parse the URL using urlparse
    parsed_url = urlparse(url)

    # Extract domain and path components
    domain = parsed_url.netloc
    path = parsed_url.path

    # Count number of subdomains
    #if isinstance(domain, bytes):
    #    domain = domain.decode('utf-8')
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
        'NoOfQMarkInURL':count_question_marks_in_url(url),
        #'DomainLength': int(len(domain)),
       # 'DomainAge': get_domain_age(domain),
       # 'HasWhois': has_whois(domain),
       # 'UseMouseOver': detect_mouseover_phishing(url),
       # 'FormCount': get_form_count(url)
    }

    return features