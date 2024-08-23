#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from utils import run_ML
from sklearn.metrics import f1_score
from itertools import groupby
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold



# In[3]:


# !pip install tldextract
# !pip install torch_geometric


# In[4]:


# data_dir = "data/URLdatasetX2_1.csv"
data_dir = "data/URLdatasetX2_1sub5.csv"
df = pd.read_csv(data_dir,index_col=0)
# n_subsample = 3000 # all
# smalldata = df.sample(n = n_subsample, random_state=1) #PC
n_subsample = 'full'; smalldata = df;
# get labels of urls
labels = smalldata.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# In[5]:


# smalldata['url']


# In[6]:


# data_dir = "data/malicious_phish_http_filter.csv"
# df = pd.read_csv(data_dir,index_col=0)
# df = df.loc[df.type!='defacement']
# # n_subsample = 3000 # all
# # smalldata = df.sample(n = n_subsample, random_state=1) #PC
# n_subsample = 'full'; smalldata = df;
# # get labels of urls
# labels = smalldata.iloc[:,-1].values
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(labels)


# In[7]:


# # data_dir = "data/LegitPhish_50_50.csv"
# # data_dir = "data/LegitPhish_80_20.csv"
# data_dir = "data/LegitPhish_90_10.csv"
# df = pd.read_csv(data_dir,index_col=0)
# n_subsample = 3000 # all
# smalldata = df.sample(n = n_subsample, random_state=1) #PC
# # smalldata = df
# # get labels of urls
# labels = smalldata.iloc[:,-1].values


# In[8]:


import collections
counter = collections.Counter(labels)
counter, len(labels)


# ### Conventional Models

# In[9]:


from utils import extract_features


# In[10]:


# Example usage:
url = "http://www.example.com/path/to/==file.html"
url_features = extract_features(url)
print(url_features)


# In[11]:


# print(url_features.keys())


# In[12]:


# get numerical and catergorical features
phish_url = []
for link in list(smalldata.iloc[:,0]):
    url_features = extract_features(link)
    phish_url.append(list(url_features.values())[1:])


# In[13]:


phish_url_df = pd.DataFrame(phish_url, columns = list(url_features.keys())[1:])


# In[14]:


# phish_url_df.head(2)


# In[15]:


phish_url_df.iloc[:,5] = pd.Categorical(phish_url_df.iloc[:,5]).codes
phish_url_df.iloc[:,8] = pd.Categorical(phish_url_df.iloc[:,8]).codes


# In[16]:


phish_url_df.head(2)


# In[17]:


# test on URLs features
run_ML(phish_url_df, labels, "URLdatasetX2", "manual")


# In[ ]:


## test on numerical URLs features
from utils import extract_numerical_features
phish_url = []
for link in list(smalldata.iloc[:,0]):
    url_features = extract_numerical_features(link)
    phish_url.append(list(url_features.values()))
run_ML(np.array(phish_url), labels, "URLdatasetX2", "manual_numerical")


# In[ ]:


np.random.seed(0)
n_samples = len(smalldata.index)
train_idx = list(np.random.choice(list(range(n_samples)), int(0.8*n_samples), replace=False))
test_idx = list(set(list(range(n_samples))).difference(set(train_idx)))
data_df = np.array(phish_url)
import lightgbm as lgb
model = lgb.LGBMClassifier(verbose=-1)
model.fit(data_df[train_idx], labels[train_idx])
y_predict=model.predict(data_df[test_idx]) 
print(f1_score(y_predict, labels[test_idx], average='macro'))


# # PyG

# ### Extract graph features from URLs for PyG

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# List of URLs
urls = list(smalldata['url'])
# Tokenization and N-grams Generation
# You can adjust ngram_range to extract different n-grams (e.g., (1, 1) for unigrams, (2, 2) for bigrams, etc.)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 4)) #5
X_counts = vectorizer.fit_transform(urls)
# TF-IDF Transformation
transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X_counts)
# Extracted Features
feature_names = vectorizer.get_feature_names_out()
X_counts_data = X_counts.toarray() # not necessary
# Train lgb
model_lgb = lgb.LGBMClassifier(verbose=-1)
model_lgb.fit(X_counts_data[train_idx], labels[train_idx])
y_predict=model_lgb.predict(X_counts_data[test_idx]) 
print(f1_score(y_predict, labels[test_idx], average='macro'))
feature_imp_gain = pd.DataFrame(sorted(zip(model_lgb.booster_.feature_importance(importance_type='gain'),
                                           feature_names), reverse=True), columns=['Value', 'Feature'])
feature_imp_split = pd.DataFrame(sorted(zip(model_lgb.booster_.feature_importance(importance_type='split'),
                                            feature_names), reverse=True), columns=['Value', 'Feature'])
top_ngrams_features = list(set(list(feature_imp_gain.iloc[:200,1]) + list(feature_imp_split.iloc[:200,1])))
cv = CountVectorizer(analyzer='char', ngram_range=(1, 4))
cv.fit(top_ngrams_features)


# In[ ]:


def extract_feature_CountVectorizer(model, url):
    return model.transform([url]).toarray().flatten()


# In[ ]:


# extract_feature_CountVectorizer(cv, urls[0])


# In[ ]:


import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError
import traceback
import logging


# In[ ]:


# return root and hyperlinks features
def get_graph_features_CountVectorizer(idx):
    url = smalldata.iloc[idx,0]
    root_feature = extract_feature_CountVectorizer(cv, url) # dict
    hyperlink_data = [list(root_feature)]
    try:    
        # find all hyperlinks
        reqs = requests.get(url, allow_redirects=False)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        urls = []
        count = 0;
        for link in soup.find_all('a'):
            # print(link.get('href'))
            weblink = link.get('href')
            if (weblink is not None) and ('http' in weblink):
                urls.append(weblink)
            count += 1
            if count > 50:
                break
        # extract numerical features in from hyperlinks
        if len(urls) > 0:
            for link in urls:
                try:
                    url_features = extract_feature_CountVectorizer(cv, link)
                    datalinkssss = list(url_features)
                    hyperlink_data.append(datalinkssss)
                except ValueError as ve:
                    # datalinkssss = list(np.zeros(15))#raw_graph_features
                    error_here = 1;
                # hyperlink_data.append(datalinkssss)
        else:
            # hyperlink_data.append(list(np.zeros(15)))#raw_graph_features
            error_here = 1;
    
    # except ConnectionError as e:
    #     # print("No rep", end = ',')
    #     # hyperlink_data.append(list(np.zeros(15))) #raw_graph_features
    #     error_here = 1; #v2
    except Exception as e:
        #logging.error(traceback.format_exc())
        error_here = 1
    
    return (idx,  hyperlink_data)


# In[ ]:


# results = [get_graph_features_CountVectorizer(i) for i in range(n_test_samples)]
# results = []
# for i in range(213, n_test_samples):
#     print(i, end =',')
#     results.append(get_graph_features_CountVectorizer(i))


# In[ ]:


import multiprocessing

multiprocessing.cpu_count()
n_cores = min(30, int(multiprocessing.cpu_count()-2))
n_cores


# In[ ]:


data_name_0123 = data_dir.split('/')[-1][:-4]
data_name_0123


# In[ ]:


data_name_0123 = data_dir.split('/')[-1][:-4]
# data_file = "data/raw_graph_features_v2.pickle" # first version 
data_file = "data/"+data_name_0123+str(n_subsample)+"_raw_graph_features_CountVectorizer.pickle" # first version 
my_file = Path(data_file)
if my_file.is_file():
    print("File exist! Load the data")
    with open(data_file, "rb") as fp:   # Unpickling
        results = pickle.load(fp)
else:
    print("File does not exist! Process the data")
    n_test_samples = int(smalldata.shape[0]) # how many link we want to test
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_cores)(delayed(get_graph_features_CountVectorizer)(i) for i in range(n_test_samples)) # test on 100 links
    with open(data_file, "wb") as fp:   #Pickling
        pickle.dump(results, fp)


# In[ ]:


data_file


# In[ ]:


len(results)


# ### Graph data class

# In[ ]:


# # Transfer data object to GPU.
# device = torch.device('cuda')
# data = data.to(device)


# In[ ]:


import torch
from torch_geometric.data import Data, Dataset

class GraphClassificationDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        # self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        # label = self.labels[idx]
        return graph
    
    def get(): pass

    def len(): pass


# ### Create dataset class for PyG

# In[ ]:


# Assume you have a list of graphs represented as Data objects and a corresponding list of labels
# Only take the url with more than 4 hyperlinks
graphs = []
labels_list = []
for i in range(len(results)):
    idx, graph_feature = results[i]
    n_hyperlinks = len(graph_feature)-1
    child_id = [i+1 for i in range(n_hyperlinks)]
    source_id = list(np.zeros(n_hyperlinks).astype(int))
    # edge_index = torch.tensor([source_id + child_id,
    #                            child_id + source_id], dtype=torch.long)
    edge_index = torch.tensor([source_id,
                               child_id], dtype=torch.long)
    x = torch.tensor(graph_feature, dtype=torch.float)
    y = torch.tensor([labels[idx]], dtype=torch.int64)
    data = Data(x=x, edge_index=edge_index, y = y)
    if n_hyperlinks > -1:
        graphs.append(data)
        labels_list.append(labels[idx])


# In[ ]:


print("Train freq: ", [len(list(group)) for key, group in groupby(sorted(labels_list))])


# In[ ]:


dataset = GraphClassificationDataset(graphs)


# In[ ]:


print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.


# In[ ]:


# dataset = dataset.shuffle()
n_samples = len(dataset)
# np.random.seed(0) 
# train_idx = list(np.random.choice(list(range(n_samples)), int(0.8*n_samples), replace=False))
# test_idx = list(set(list(range(n_samples))).difference(set(train_idx)))
# train_idx[:10]


# In[ ]:


# train_dataset = dataset[:int(0.8*n_samples)]
# test_dataset = dataset[int(0.8*n_samples):]
train_dataset = [dataset[idx] for idx in train_idx]
test_dataset = [dataset[idx] for idx in test_idx]


# In[ ]:


len(train_dataset), len(test_dataset)


# In[ ]:


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ### Build and train PyG

# In[ ]:


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GCN(torch.nn.Module):
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
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


# In[ ]:


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # print(data.x.shape)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    correct = 0
    # for data in loader:  # Iterate in batches over the training/test dataset.
    #     out = model(data.x, data.edge_index, data.batch)  
    #     pred = out.argmax(dim=1)  # Use the class with highest probability.
    #     correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    # return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    true_labels = []
    pred_labels = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        true_labels += data.y.tolist()
        pred_labels += pred.tolist()
        # print(pred_labels)
    return f1_score(true_labels, pred_labels, average='macro')

for epoch in range(0, 20):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Train F1: {train_acc:.4f}, Test F1: {test_acc:.4f}')


# In[ ]:


# ['kNN', 'LightGBM'] min, max, avg of child features [0.82 0.92]


# ### GNN for dim. reduction

# #### Net 1: 0.87, 0.88

# In[ ]:


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCNdimReduce(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCNdimReduce, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        
        return x
    
    def dimReduce(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # # 3. Apply a final classifier
        x = self.lin1(x)
        return x


# #### Net 2

# In[ ]:


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCNdimReduceV2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCNdimReduceV2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
       
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        # 3. Apply a final classifier
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        
        return x
    
    def dimReduce(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x


# #### Test model

# In[ ]:


n_hidden_channels = 16
model = GCNdimReduce(hidden_channels=n_hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # print(data.x.shape)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    correct = 0
    true_labels = []
    pred_labels = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        true_labels += data.y.tolist()
        pred_labels += pred.tolist()
    return f1_score(true_labels, pred_labels, average='macro')

for epoch in range(0, 10):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Train F1: {train_acc:.4f}, Test F1: {test_acc:.4f}')


# #### Def

# In[ ]:


phish_url_vectorizer = []
for link in list(smalldata.iloc[:,0]):
    url_features = extract_feature_CountVectorizer(cv, link)
    phish_url_vectorizer.append(list(url_features))
# run_ML(concatGNN, labels, "URLdatasetX2", "concatGNN")


# In[ ]:


def train_model(all_data_loader, train_loader, test_loader, n_hidden_channels = 16, n_epoch=1, lr=0.001):
    # n_hidden_channels = 16
    model = GCNdimReduce(hidden_channels=n_hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            # print(data.x.shape)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()
        correct = 0
        true_labels = []
        pred_labels = []
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            true_labels += data.y.tolist()
            pred_labels += pred.tolist()
        return f1_score(true_labels, pred_labels, average='macro')

    for epoch in range(0, n_epoch):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Train F1: {train_acc:.4f}, Test F1: {test_acc:.4f}')
    
    model.eval()
    dim_vec = torch.empty((0, n_hidden_channels), dtype=torch.float32)
    for data in all_data_loader:
        dim_x = model.dimReduce(data.x, data.edge_index, data.batch)
        dim_vec = torch.cat((dim_vec, dim_x), 0)
    return (dim_vec)


# In[ ]:


all_data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
dim_vec = train_model(all_data_loader, train_loader, test_loader)
# model.eval()
# dim_vec = torch.empty((0, n_hidden_channels), dtype=torch.float32)
# # data = next(iter(test_loader))
# for data in all_data_loader:
#     dim_x = model.dimReduce(data.x, data.edge_index, data.batch)
#     dim_vec = torch.cat((dim_vec, dim_x), 0)


# In[ ]:


# concatGNN = np.concatenate((np.array(phish_url_vectorizer), dim_vec.detach().numpy()),axis=1)
# model_lgb = lgb.LGBMClassifier(verbose=-1)
# model_lgb.fit(concatGNN[train_idx], labels[train_idx])
# y_predict=model_lgb.predict(concatGNN[test_idx]) 
# print(f1_score(y_predict, labels[test_idx], average='macro'))


# In[ ]:


# model_lgb = lgb.LGBMClassifier(verbose=-1)
# graph_embedding = dim_vec.detach().numpy()
# model_lgb.fit(graph_embedding[train_idx], labels[train_idx])
# y_predict=model_lgb.predict(graph_embedding[test_idx]) 
# print(f1_score(y_predict, labels[test_idx], average='macro'))


# In[ ]:


# Stats graph feature
idx, vec = results[0]; vec = np.array(vec); n_features_counter = int(vec.shape[1]);
hyperlink_features = np.zeros((smalldata.shape[0], 3*n_features_counter))
for idx, hyper_np in results:
    # print(idx, hyper_np)
    hyper_np = np.array(hyper_np)
    if hyper_np.shape[0] >= 2:
        hyperlink_features[idx, :] = np.hstack((hyper_np.min(axis=0),hyper_np.max(axis=0), hyper_np.mean(axis=0)))
    # hyperlink_features[idx, :] = hyper_np


# In[ ]:


concatGNN_graph = np.concatenate((np.array(phish_url_vectorizer), hyperlink_features),axis=1)
# concatGNN_graph = np.concatenate((np.array(phish_url_vectorizer), hyperlink_features,  dim_vec.detach().numpy()),axis=1)


# In[ ]:


model_lgb2 = lgb.LGBMClassifier(verbose=-1)
model_lgb2.fit(concatGNN_graph[train_idx], labels[train_idx])
y_predict=model_lgb2.predict(concatGNN_graph[test_idx]) 
print(f1_score(y_predict, labels[test_idx], average='macro'))


# In[ ]:


def runGRAPHISH(train_idx, test_idx):
    stack_GNNs_graph = np.concatenate((np.array(phish_url_vectorizer), hyperlink_features),axis=1)
    for i in range(10):
        np.random.seed(i) 
        # train_idx_new = list(np.random.choice(list(range(n_samples)), int(0.8*n_samples), replace=False))
        # train_idx_new = list(set(train_idx_new).difference(test_idx))
        train_idx_new = list(np.random.choice(train_idx, int(0.8*len(train_idx)), replace=False))
        print(train_idx_new[:5])
        train_dataset_new = [dataset[idx] for idx in train_idx_new]
        train_loader_new = DataLoader(train_dataset_new, batch_size=8, shuffle=True)
        n_hidden_channels = 2
        n_epoch = 1
        dim_vec_new = train_model(all_data_loader, train_loader_new, test_loader, n_hidden_channels, n_epoch)
        stack_GNNs_graph = np.concatenate((stack_GNNs_graph, dim_vec_new.detach().numpy()),axis=1)
    model_lgb2 = lgb.LGBMClassifier(verbose=-1)
    model_lgb2.fit(stack_GNNs_graph[train_idx], labels[train_idx])
    y_predict=model_lgb2.predict(stack_GNNs_graph[test_idx]) 
    y_proba=model_lgb2.predict_proba(stack_GNNs_graph[test_idx])[:,1]
    print(f1_score(y_predict, labels[test_idx], average='macro'))
    df_results = pd.DataFrame({'true_label': labels[test_idx], 'pred_label': y_predict, 'predict_proba': y_proba})
    return df_results


# In[ ]:


runGRAPHISH(train_idx, test_idx)


# In[ ]:


n_loops = 1; n_folds = 5;
base_dir = 'comparision_results/sub5prob'
approach = 'GRAPHISH'
data_set = data_dir[5:-4]
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
for i in range(n_loops):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state = i)
    for fold, (train_idx, test_idx) in enumerate(cv.split(phish_url_vectorizer)):
        path_dir = base_dir +'/' + data_set + '_run_'+str(i)+'_'+ 'fold_'+str(fold)+'_'+approach
        print('Run: ', i, ', fold: ', fold)
        # X_train = X[train_idx]
        # X_test = X[test_idx]
        # y_train = y[train_idx]
        # y_test = y[test_idx]
        df = runGRAPHISH(train_idx, test_idx)
        path_dir = base_dir +'/' + data_set + '_run_'+str(i)+'_'+ 'fold_'+str(fold)+'_'+approach
        df.to_csv(path_dir + '_GRAPHISH_labels.csv', index=False)

