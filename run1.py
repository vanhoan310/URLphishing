
import numpy as np
import pandas as pd
from utils import run_ML


data_dir = "data/URLdatasetX2_1.csv"
df = pd.read_csv(data_dir,index_col=0)


smalldata = df.sample(n = 1000, random_state=1)
# smalldata = df



if 'PhiUSIIL' in data_dir:
    urldata = np.array(smalldata[['URLLength', 'DomainLength', 'IsDomainIP', \
           'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',\
           'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation',\
           'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',\
           'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',\
           'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',\
           'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS']])
    labels = smalldata.iloc[:,-1].values
    run_ML(urldata, labels, "PhiUSIILmock", "standard")
if 'URLdatasetX2' in data_dir:
    labels = smalldata.iloc[:,-1].values
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)


# In[ ]:


# labels


# ### Feature extraction

# In[ ]:


from utils import extract_features


# In[ ]:


# Example usage:
url = "http://www.example.com/path/to/==file.html"
url_features = extract_features(url)
print(url_features)


# In[ ]:


# print(url_features.keys())


# In[ ]:


phish_url = []
for link in list(smalldata.iloc[:,0]):
    url_features = extract_features(link)
    phish_url.append(list(url_features.values())[1:])



phish_url_df = pd.DataFrame(phish_url, columns = list(url_features.keys())[1:])


#
phish_url_df.iloc[:,5] = pd.Categorical(phish_url_df.iloc[:,5]).codes
phish_url_df.iloc[:,8] = pd.Categorical(phish_url_df.iloc[:,8]).codes


# In[ ]:



run_ML(phish_url_df, labels, "URLdatasetX2", "manual")

