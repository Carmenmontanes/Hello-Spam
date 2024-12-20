#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("D:/Master/Master Dissertation/Dissertation/cleaned_spamdata.csv")

# Preprocess text
df['v2'] = df['v2'].fillna('').str.lower()
X_text = df['v2']  
y = df['v1']  

# Feature Engineering
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), max_df=0.95, min_df=0.01)
X_tfidf = vectorizer.fit_transform(X_text).toarray()

# Resampling to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=5)

# Define classifiers
extra_trees = ExtraTreesClassifier(n_estimators=1500, random_state=50)
random_forest = RandomForestClassifier(n_estimators=1500, random_state=50, class_weight='balanced')

voting_classifier = VotingClassifier(estimators=[
    ('extra_trees', extra_trees), 
    ('random_forest', random_forest)
], voting='soft')

# Train model
voting_classifier.fit(X_train, y_train) 

# Predictions
y_pred = voting_classifier.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

