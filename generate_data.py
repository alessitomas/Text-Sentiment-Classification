import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

df_data = pd.read_csv("./data/training.1600000.processed.noemoticon.csv", sep=",", encoding="latin1", names=["label", "id", "datetime", "flag", "user", "text"])

# Step 1: Remove unnecessary fields
df_data = df_data[['label', 'text']]

df_data.dropna(subset=['text', 'label'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df_data['text'] = df_data['text'].apply(preprocess_text)

import re
def remove_mentions_and_links(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return text

df_data['text'] = df_data['text'].apply(remove_mentions_and_links)

def remove_punctuation(text):
    text = re.sub(r'[^\w\s!?]', '', text)
    return text


df_data['text'] = df_data['text'].apply(remove_punctuation)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Pipeline 1: Naive Bayes with CountVectorizer
pipeline_nb_cv = Pipeline([
    ('vectorizer', CountVectorizer()),  # Bag of words
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])


# Pipeline 2: Naive Bayes with TfidfVectorizer
pipeline_nb_tfidf = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TF-IDF
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

# Pipeline 3: Logistic Regression with CountVectorizer
pipeline_lr_cv = Pipeline([
    ('vectorizer', CountVectorizer()),  # Bag of words
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression
])



pipeline_lr_tfidf = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TF-IDF
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression
])


pipelines = {
    'Naive Bayes + CountVectorizer': pipeline_nb_cv,
    'Naive Bayes + TfidfVectorizer': pipeline_nb_tfidf,
    'Logistic Regression + CountVectorizer': pipeline_lr_cv,
    'Logistic Regression + TfidfVectorizer': pipeline_lr_tfidf
}

data = {
    'Naive Bayes + CountVectorizer' : {"acc_train": [], "acc_test": []},
    'Naive Bayes + TfidfVectorizer': {"acc_train": [], "acc_test": []},
    'Logistic Regression + CountVectorizer': {"acc_train": [], "acc_test": []},
    'Logistic Regression + TfidfVectorizer': {"acc_train": [], "acc_test": []}
}


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def generate_learning_curve(df, text_column, label_column, pipeline, data, fractions=np.linspace(0.1, 1.0, 10), repeats=10, test_size=0.2, random_state=42):

    
    for frac in fractions:
        train_acc = []
        test_acc = []
        
        # Resample and shuffle the dataset
        df_sampled = df.sample(frac=frac, random_state=random_state).sample(frac=1)  # Resample and shuffle
        X = df_sampled[text_column]
        y = df_sampled[label_column]
        
        # Repeat multiple times for more stable accuracy
        for _ in range(repeats):
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Train the model on the training data
            pipeline.fit(X_train, y_train)
            
            # Get accuracy for training and testing sets
            train_acc.append(accuracy_score(y_train, pipeline.predict(X_train)))
            test_acc.append(accuracy_score(y_test, pipeline.predict(X_test)))
        
        # Store mean accuracies for the current fraction
        data["acc_train"].append(np.mean(train_acc))
        data["acc_test"].append(np.mean(test_acc))



for name, pipeline in pipelines.items():
    print(name)
    generate_learning_curve(df_data, "text", "label", pipeline, data[name])


import json

# Save the dictionary to a file
with open('full_data.json', 'w') as file:
    json.dump(data, file, indent=4)