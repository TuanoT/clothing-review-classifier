import re
import numpy as np

# Load stopwords
with open("data/stopwords_en.txt", "r") as f:
    stopwords = set(line.strip() for line in f)

def tokenize_review(review: str) -> list:
    """
    Tokenizes a review string into a list of words, removing punctuation and converting to lowercase.
    
    Args:
        review (str): The review text to tokenize.
        
    Returns:
        list: A list of tokens (words) from the review.
    """
    tokens = re.findall(r'\b\w+\b', review.lower())
    for token in tokens:
        if token in stopwords:
            tokens.remove(token)

    return tokens

def get_mean_token_vector(title_tokens: list, review_tokens: list, model) -> list:
    """
    Generates a mean vector for the tokens in a review using a trained Word2Vec model.
    Args:
        title_tokens (list): List of tokens from the review title.
        review_tokens (list): List of tokens from the review text.
        model: A trained Word2Vec model.
    Returns:
        list: A mean vector representing the review.
    """
    # Get a list of token vectors for each token in the title
    title_vectors = []
    for token in title_tokens:
        if token in model.wv:
            vector = model.wv[token]
            title_vectors.append(vector)

    # Get the mean vector for the title
    if title_vectors:
        mean_title_vector = np.mean(title_vectors, axis=0)
    else:
        mean_title_vector = np.zeros(model.vector_size)
    
    # Get a list of token vectors for each token in the review
    review_vectors = []
    for token in review_tokens:
        if token in model.wv:
            vector = model.wv[token]
            review_vectors.append(vector)

    # Get the mean vector for the review
    if review_vectors:
        mean_review_vector = np.mean(review_vectors, axis=0)
    else:
        mean_review_vector = np.zeros(model.vector_size)
    
    return np.concatenate((mean_title_vector, mean_review_vector))
