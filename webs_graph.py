# for url in list(smalldata.iloc[:,0]):
def get_features(idx):
    url = smalldata.iloc[idx,0]
    # url = 'https://polarklimatsgserver.blogspot.com/'
    try:    
        reqs = requests.get(url)
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
        # extract numerical features in link
        if len(urls) > 0:
            hyperlink_data = []
            for link in urls:
                try:
                    url_features = extract_numerical_features(link)
                    datalinkssss = list(url_features.values())
                except ValueError as ve:
                    datalinkssss = list(np.zeros(15))
                hyperlink_data.append(datalinkssss)
            hyperlink_data = np.array(hyperlink_data)
            hyper_np = np.hstack((np.array([len(urls), count, float(len(urls))/(count + 1)]),
                                  hyperlink_data.min(axis=0),hyperlink_data.max(axis=0), hyperlink_data.mean(axis=0)))
        else:
            hyper_np = np.hstack((np.array([len(urls), count, float(len(urls))/(count + 1)]),np.zeros(45)))
    
    except ConnectionError as e:
        # print("No rep", end = ',')
        hyper_np = np.zeros(48)
    # hyperlink_features[enum, :] = hyper_np
    # print(enum, end =',')
    # enum += 1
    return (idx, hyper_np)

from joblib import Parallel, delayed
results = Parallel(n_jobs=2)(delayed(get_features)(i) for i in range(len(smalldata.index)))