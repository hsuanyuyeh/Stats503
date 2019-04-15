
# coding: utf-8

# In[ ]:


import os
os.chdir('/Users/yehhsuan-yu/Umich/Stats503/project')
#os.chdir('/home/hsuanyu')
train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
df=train.append(test,ignore_index=True)
text=df["comment_text"]

word_vectorizer=CountVectorizer(
    #sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='word',
    #token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=30)
# tokenize and build vocab
word_vectorizer.fit(text)


train_features = word_vectorizer.transform(train['comment_text'])
test_features=word_vectorizer.transform(test['comment_text'])
print(train_features.shape)


# In[ ]:


# PCA
from sklearn.decomposition import PCA
PCA_tfidf = PCA(n_components=30).fit(train_features.toarray())
import pickle
os.chdir('/Users/yehhsuan-yu/Umich/Stats503/project')
#os.chdir('/home/hsuanyu')
pickle.dump(PCA_tfidf, open('PCA30_count.p','wb'))


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
df=train.append(test,ignore_index=True)
text=df["comment_text"]

word_vectorizer_tfidf=TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='word',
    #token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=30)
word_vectorizer_tfidf.fit(text)

train_features_tfidf = word_vectorizer_tfidf.transform(train['comment_text'])
test_features_tfidf = word_vectorizer_tfidf.transform(test['comment_text'])
print(train_features_tfidf.shape)


# In[ ]:


# PCA
PCA_tfidf = PCA(n_components=30).fit(train_features_tfidf.toarray())
import pickle
os.chdir('/Users/yehhsuan-yu/Umich/Stats503/project')
#os.chdir('/home/hsuanyu')
pickle.dump(PCA_tfidf, open('PCA30_tfidf.p','wb'))


# In[4]:


# KNN
#from sklearn.neighbors import KNeighborsClassifier

#myList = list(range(1,50))
#neighbors = filter(lambda x: x%2 != 0, myList)
#cv_scores = []
#for k in neighbors:
#    knn_tfidf = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
#    scores = cross_val_score(knn_tfidf, train_features_tfidf.toarray(), np.array(train['toxic']), cv=5, scoring='accuracy')
#    cv_scores.append(scores.mean())

#MSE = [1 - x for x in cv_scores]
#optimal_k = neighbors[MSE.index(min(MSE))]
#print ("The optimal number of neighbors is %d" % optimal_k)


# In[ ]:


#predict_tfidf = knn_tfidf.predict(test_features_tfidf.toarray())
#predict_tfidf = np.array(predict_tfidf)
#print(sum(predict_tfidf == test['toxic'])/31915.0)

