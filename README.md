# Twitter Sentiment Analysis
## Author: Mohan Krishna K
## Overview
This project focuses on sentiment analysis of Twitter data, classifying tweets into positive or negative categories using Natural Language Processing (NLP) techniques and machine learning models. The objective is to preprocess tweet data and build models to accurately predict the sentiment of new tweets.

## Dataset
The dataset used for this project consists of labeled Twitter data and the link is here https://www.kaggle.com/karan842/twitter-sentimental-analysis/data

### Project Overview
This project analyzes the sentiment of tweets, determining whether a tweet is positive or negative. We use text preprocessing techniques and machine learning models to classify the sentiment of tweets.
![image](https://github.com/user-attachments/assets/241c547b-1079-41e5-880d-33ce58cacec4)


## Goals
Clean the tweet text for better analysis.
Extract features from the cleaned text using TF-IDF (Term Frequency-Inverse Document Frequency).
Train models to predict tweet sentiment.
Evaluate the models to determine which one performs the best.
Key Steps

### 1. Text Preprocessing
Before we can use the tweets for machine learning, we clean the text by:

Converting to lowercase.
Replacing usernames and URLs with placeholders.
Removing unnecessary characters and stopwords (common words like "and" or "the").
Handling emojis and converting them to text (e.g., ":)" becomes "smile").
Lemmatizing words to convert them to their root form (e.g., "running" becomes "run").

### 2. Feature Extraction
We use TF-IDF vectorization to convert the cleaned tweets into numerical data that the machine learning models can understand.

### 3. Machine Learning Models
We trained three machine learning models:

Bernoulli Naive Bayes: A simple and fast model.
Linear Support Vector Classification (LinearSVC): A model that works well with text data.
Logistic Regression: The best-performing model in this project.

### 4. Model Evaluation
We evaluated the models using accuracy and precision. The Logistic Regression model performed the best, achieving an accuracy of 82%, while Bernoulli Naive Bayes was the fastest to train.

## Conclusion
This project shows how tweets can be analyzed using machine learning to determine sentiment. The Logistic Regression model provided the best balance of accuracy and performance, making it ideal for sentiment analysis tasks.

Files
vectoriser-ngram-(1,2).pickle: Saved vectorizer for feature extraction.
Sentiment-LR.pickle: Saved Logistic Regression model.
Sentiment-BNB.pickle: Saved Bernoulli Naive Bayes model.
Requirements
Python 3.x
Libraries: numpy, pandas, scikit-learn, nltk, matplotlib, seaborn, wordcloud
