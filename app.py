#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-16T07:01:19.635Z
"""

import pandas as pd
import numpy as np
#pandas is used for handling tabular data (like CSV files).
#numpy is used for numerical operations (arrays, math, etc).



df_fake = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ML mini project FAKE NEWS DETECTION\Fake.csv")
df_true = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ML mini project FAKE NEWS DETECTION\True.csv")


df_fake.head()


df_fake["label"] = 0
df_true["label"] = 1
#Adds a new column label:
#0 for fake news
#1 for real news
#This helps later for classification (supervised learning needs labeled data).



df_true = df_true[['text','label']]
df_fake = df_fake[['text','label']]

df_concat = pd.concat([df_fake, df_true], axis=0)
#Combines both fake and true news into a single DataFrame df.
#axis=0 stacks them vertically (one below the other).
df_concat

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    if isinstance(text, str):  # make sure it's a string
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # remove punctuation/special characters
        text = text.lower()                           # convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply to a specific column (e.g., 'text_column') in df_concat
df_concat['cleaned_text'] = df_concat['text'].apply(clean_text)

# View the cleaned column
df=df_concat[[ 'cleaned_text','label']]
print(df)


df.shape

df.isnull().sum() # no null values

df['label'].value_counts()

df_true.shape # true news

df_fake.shape # fake news

df = df.sample(frac=1)
#df.sample(frac=1) randomly shuffles all rows in the DataFrame.
#This prevents the model from learning in the order the data was added (first all fake, then all true).
#frac=1 means "sample 100% of the rows, in random order".

df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)
#After shuffling, the old row index is no longer meaningful, so:
#reset_index() assigns new indices (0 to N).
#drop(["index"], axis=1) removes the old index column from the DataFrame.

df.head()


X = df['cleaned_text']
y = df['label']
#X is your feature data (news content).
#y is your label (0 = fake, 1 = real).
#This prepares the data for train-test splitting.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#Splits the dataset into 80% training and 20% testing.
#random_state=0 ensures the same split every time (for reproducibility).



from sklearn.feature_extraction.text import TfidfVectorizer
#Imports TfidfVectorizer, a tool that converts text into numeric vectors based on word frequency and importance (TF-IDF).
#This is essential for feeding text into machine learning models.

vectorizer = TfidfVectorizer()
vec_train = vectorizer.fit_transform(X_train)
vec_test = vectorizer.transform(X_test)
#TfidfVectorizer() creates the vectorizer.
#.fit_transform(X_train):
#Learns vocabulary from the training data.
#Converts training text into sparse numerical vectors.
#.transform(X_test):
#Applies the same vocabulary to the test data (no learning here)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(vec_train, y_train)
#Imports and initializes a Multinomial Naive Bayes classifier — very effective for text classification tasks.
#.fit(vec_train, y_train) trains the model using the TF-IDF vectors and their labels.



y_pred = clf.predict(vec_test)
#Uses the trained model to make predictions (y_pred) on the test set (vec_test).
#These are the predicted labels: 0 for fake, 1 for real.

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

#Imports a metric to calculate the model’s accuracy.
#accuracy_score(y_test, y_pred) compares the predicted labels with the true test labels.
#The result tells you how well your model is performing on unseen data.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(vec_train, y_train)
y_pred_lr = log_reg.predict(vec_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nb = MultinomialNB()
nb.fit(vec_train, y_train)
y_pred_nb = nb.predict(vec_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svm = LinearSVC(dual='auto')# explicitly set to avoid warning
svm.fit(vec_train, y_train)
y_pred_svm = svm.predict(vec_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=50,random_state=42)
rf.fit(vec_train, y_train)
y_pred_rf = rf.predict(vec_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


import warnings
warnings.filterwarnings("ignore")


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(vec_train, y_train)
y_pred_xgb = xgb.predict(vec_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
#confusion_matrix: shows the number of:
#True Positives (real news correctly predicted as real)
#True Negatives (fake predicted as fake)
#False Positives (fake predicted as real)
#False Negatives (real predicted as fake)
#classification_report gives:
#Precision: How many predicted positives were actually correct
#Recall: How many actual positives were caught
#F1-score: Harmonic mean of precision and recall
#Support: Number of true instances for each label




sample = ["President announces new policy to fight inflation."]
sample_vec = vectorizer.transform(sample)
print(clf.predict(sample_vec))

#Defines a custom news headline or text.
#Transforms it into TF-IDF form using the previously fitted vectorizer.
#Passes it to the model to predict: 0 = fake, 1 = real.



samples = [
    "NASA discovers water on Mars",
    "Click here to win free iPhones",
    "Government collapses under pressure"
]

samples_vec = vectorizer.transform(samples)
print(clf.predict(samples_vec))
#Tests the model on multiple new inputs.
#Again, transforms the input using the same vectorizer.
#Outputs an array of predictions (0 or 1) for each sentence.

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)

# Print text outputs
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

# 🔹 Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Fake", "Real"], 
            yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


import joblib

joblib.dump(svm, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")