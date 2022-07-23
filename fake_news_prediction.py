#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('stopwords')

#DATASET
news_dataset.head()

#STATS
news_dataset.shape
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')
news_dataset['data'] = news_dataset['author']+' '+news_dataset['title']
print(news_dataset['data'])
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
print(X,'\n\n\n\n',Y)

#STEMMING
stem_words = PorterStemmer()
def stemming(data):
    stemmed_content = re.sub('[^a-zA-Z]',' ',data)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stem_words.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
 news_dataset['data'] = news_dataset['data'].apply(stemming)

#VISUALIZING
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords,
                        max_words=200, 
                        random_state=42).generate(str(news_dataset['data']))

plt.figure(figsize=(15,10))
plt.axis("off")
plt.title("Words frequented in text", fontsize=15)
plt.imshow(wordcloud.recolor(colormap= 'viridis' , random_state=42), alpha=0.98)
plt.show()
print(news_dataset['data'])

#MODELLING
X = news_dataset['data'].values
Y = news_dataset['label'].values
print(X,'\n\n\n\n',Y)
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

#ACCURACY CHECK
 #TRAINING ACCURACY
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
 #TESTING ACCURACY
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

#MODEL PREDICTION
X_new = X_test[0]
prediction = model.predict(X_new)

if (prediction[0]==0):
    print('The news is Real')
else:
    print('The news is Fake')
print(Y_test[0])

