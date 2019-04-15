
# coding: utf-8

# In[4]:


import os
import pandas as pd
os.chdir('/Users/yehhsuan-yu/Umich/Stats503/project')
#os.chdir('/home/hsuanyu')
train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')


# # CountVectorizer

# In[5]:


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


# KNN
from sklearn.neighbors import KNeighborsClassifier

myList = list(range(1,50))
neighbors = filter(lambda x: x%2 != 0, myList)
cv_scores = []
for k in neighbors:
    knn_count = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
    scores = cross_val_score(knn_count, train_features.toarray(), np.array(train['toxic']), cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE_count = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)
MSE_count_df = pd.DataFrame(MSE_count)
MSE_count_df.to_csv('/home/hsuanyu/MSE_count.csv')


# In[ ]:


#predict_tfidf = knn_tfidf.predict(test_features_tfidf.toarray())
#predict_tfidf = np.array(predict_tfidf)
#print(sum(predict_tfidf == test['toxic'])/31915.0)


# # TfidfVectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
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


# KNN
from sklearn.neighbors import KNeighborsClassifier

myList = list(range(1,50))
neighbors = filter(lambda x: x%2 != 0, myList)
cv_scores = []
for k in neighbors:
    knn_tfidf = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
    scores = cross_val_score(knn_tfidf, train_features_tfidf.toarray(), np.array(train['toxic']), cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE_tfidf = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)
MSE_tfidf_df = pd.DataFrame(MSE_tfidf)
MSE_tfidf_df.to_csv('/home/hsuanyu/MSE_tfidf.csv')


# In[ ]:


#predict_tfidf = knn_tfidf.predict(test_features_tfidf.toarray())
#predict_tfidf = np.array(predict_tfidf)
#print(sum(predict_tfidf == test['toxic'])/31915.0)

