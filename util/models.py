"""
This module generates machine learning models for review classification.
"""

# Import necessary libraries
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import ast
import joblib
import os
from util.reviewprocessor import tokenize_review, get_mean_token_vector

def generate_models(test_mode: bool=False):
    """
    Generates machine learning models for review classification using Word2Vec features.
    Args:
        test_mode (bool): If True, runs the model training and cross-validation.
    """

    # Load the data
    df = pd.read_csv('data/processed.csv')
    df["Review Tokens"] = df["Review Tokens"].apply(ast.literal_eval)

    # Create title tokens for each review
    df["Title Tokens"] = df["Title"].apply(tokenize_review)

    # Combine review tokens and title tokens into a single column
    df["Tokens"] = df["Title Tokens"] + df["Review Tokens"]
    print("\nTokens generated successfully.")

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=df["Tokens"], vector_size=200,
                     window=5, min_count=1, workers=4)
    # Save the Word2Vec model
    os.makedirs('models', exist_ok=True)
    word2vec_model.save('models/word2vec.model')
    print("\nWord2Vec model trained and saved to models/word2vec.model.")

    # Generate Word2Vec features
    X = []
    for _, row in df.iterrows():
        title_tokens = row["Title Tokens"]
        review_tokens = row["Review Tokens"]
        mean_vector = get_mean_token_vector(title_tokens, review_tokens, word2vec_model)
        X.append(mean_vector)
    X = np.array(X)
    print(f"\nWord2Vec features generated successfully. Vector shape: {X.shape}")

    # Target variable
    y = df["Recommended IND"].values

    if test_mode:
        # Test Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier()
        print("\nDecision Tree Classifier fit successfully.")
        print("Starting cross-validation...")
        scores = cross_val_score(dt_classifier, X, y, cv=5)
        print("\nCross-validation done.")
        print("\nDecision Tree scores:", scores)

        # Test Logistic Regression Classifier
        log_classifier = LogisticRegression(max_iter=1000)
        print("\nLogistic Regression Classifier fit successfully.")
        print("Starting cross-validation...")
        log_scores = cross_val_score(log_classifier, X, y, cv=5)
        print("\nCross-validation done.")
        print("\nLogistic Regression:", log_scores)

        # Test K-Nearest Neighbors Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        print("\nK-Nearest Neighbors Classifier fit successfully.")
        print("Starting cross-validation...")
        knn_scores = cross_val_score(knn_classifier, X, y, cv=5)
        print("\nCross-validation done.")
        print("\nK-Nearest Neighbors scores:", knn_scores)

    # Save the Logistic Regression model
    os.makedirs('models', exist_ok=True)
    log_classifier = LogisticRegression(max_iter=1000)
    log_classifier.fit(X, y)
    joblib.dump(log_classifier, 'models/review_classifier.pkl')
    print("\nmodels/review_classifier.pkl saved successfully.\n")