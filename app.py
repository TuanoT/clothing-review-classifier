from flask import Flask, render_template, request, redirect, url_for
from gensim.models import Word2Vec
from util.preprocessor import generate_csv
from util.models import generate_models
from util.reviewprocessor import tokenize_review, get_mean_token_vector
from nltk.stem import PorterStemmer
import pandas as pd
import csv
import os
import joblib

# Create the Flask application
app = Flask(__name__)

# Load the clothing data
if not os.path.exists('data/clothes.csv'): generate_csv()
clothes_df = pd.read_csv('data/clothes.csv')

# Load the Models
if not os.path.exists('models/word2vec.model') or not os.path.exists('models/review_classifier.pkl'):
    generate_models()
vector_model = Word2Vec.load('models/word2vec.model')
classifier = joblib.load('models/review_classifier.pkl')
stemmer = PorterStemmer()

# Search Items Page
@app.route('/', methods=['GET'])
def index():

    # Get the items dict based on the search query
    query_string = request.args.get('search', '')
    items = get_filter_items_dict(query_string)
    items_found = len(items)

    # Only render the first 20 items
    items = items[:20]

    return render_template('index.html', items=items, query_string=query_string, items_found=items_found)


# Item details and reviews page
@app.route('/item/<int:item_id>')
def item(item_id):

    # Get the items DataFrame row based on the item_id
    row = clothes_df[clothes_df['Clothing ID'] == item_id]
    if row.empty:
        return "Clothing item not found.", 404
    
    # Convert the row to a dictionary
    columns = ['Clothing ID', 'Division Name', 'Department Name', 'Class Name', 'Clothes Title', 'Clothes Description']
    item = row[columns].iloc[0].to_dict()

    # Load reviews into a dataframe
    reviews_df = pd.DataFrame()
    if os.path.exists('data/clothing-reviews.csv'):
        reviews_df = pd.read_csv('data/clothing-reviews.csv', header=None,
                                  names=['Review ID', 'Clothing ID', 'Title', 'Text', 'Recommended IND'])
        # Reverse the order of reviews and filter by item_id
        reviews_df = reviews_df[reviews_df['Clothing ID'] == item_id].tail(5).iloc[::-1]

    # Convert reviews to a dictionary
    if not reviews_df.empty:
        reviews = reviews_df[['Review ID', 'Text', 'Title', 'Recommended IND']].to_dict(orient='records')
    else:
        reviews = []

    return render_template('item.html', item=item, reviews=reviews)


# Submit a review for an item
@app.route('/submit_review/<int:item_id>', methods=['POST'])
def submit_review(item_id):

    review_title = request.form.get('title', '').strip()
    review_text = request.form.get('review', '').strip()

    # Predict if the review is recommended or not
    title_tokens = tokenize_review(review_title)
    review_tokens = tokenize_review(review_text)
    mean_vector = get_mean_token_vector(title_tokens, review_tokens, vector_model)
    x = mean_vector.reshape(1, -1)
    recommended = int(classifier.predict(x))

    # Get the current number of reviews to assign a review_id
    try:
        with open('data/clothing-reviews.csv', 'r') as f:
            review_id = sum(1 for _ in f)
    except FileNotFoundError:
        review_id = 0

    # Save the review to a CSV file with the new review_id
    with open('data/clothing-reviews.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([review_id, item_id, review_title, review_text, recommended])

    # Update the rating of the clothing item
    update_clothes_rating(item_id)

    return redirect(url_for('item', item_id=item_id))


# Toggle recommendation state for a review
@app.route('/toggle_recommendation', methods=['POST'])
def toggle_recommendation():

    # Get the review_id and new recommended state from the form
    review_id = int(request.form['review_id'])
    recommended = int(request.form['recommended'])
    recommended = 1 if recommended == 0 else 0

    rows = []
    item_id = -1  # Initialize item_id to -1 to handle cases where no item is found

    # Read all rows and update the recommended field for the matching review_id
    with open('data/clothing-reviews.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) == review_id:
                row[-1] = str(recommended)
                item_id = int(row[1])  # Store the item_id for redirection
            rows.append(row)

    # Write updated rows back into the CSV file
    with open('data/clothing-reviews.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Update the rating of the clothing item
    update_clothes_rating(item_id)

    # Redirect back to the item's page
    return redirect(url_for('item', item_id=item_id))


def get_filter_items_dict(query: str) -> dict:
    """
    Filters the clothing items based on the search query and sorts by rating.
    Args:
        query (str): The search query to filter clothing items.
    Returns:
        dict: A dictionary of filtered clothing items sorted by rating.
    """
    # Load the latest clothes data with ratings
    clothes_df = pd.read_csv('data/clothes.csv')

    # Tokenize and stem the query
    query = query.lower()
    query_tokens = tokenize_review(query)
    query_tokens = [stemmer.stem(token) for token in query_tokens]

    # Function to match rows based on the query tokens
    def match_row(search_text):
        text_tokens = tokenize_review(search_text)
        text_tokens = [stemmer.stem(token) for token in text_tokens]
        return any(t in text_tokens for t in query_tokens)

    # Filter the DataFrame based on the query
    if query != '':
        filtered_df = clothes_df[clothes_df['Search Text'].apply(match_row)]
    else:
        filtered_df = clothes_df
    filtered_df = filtered_df.sort_values(by='Rating', ascending=False)

    return filtered_df[['Clothes Title', 'Clothes Description', 'Clothing ID', 'Rating']].to_dict(orient='records')


def update_clothes_rating(clothing_id: int):
    """
    Updates the rating of a clothing item based on its reviews.
    Args:
        clothing_id (int): The ID of the clothing item to update.
    """
    # Load reviews and clothes data
    reviews_df = pd.read_csv('data/clothing-reviews.csv', header=None,
                              names=['Review ID', 'Clothing ID', 'Title', 'Text', 'Recommended IND'])
    clothes_df = pd.read_csv('data/clothes.csv')

    # Filter reviews by clothing ID and calculate the rating
    item_reviews = reviews_df[reviews_df['Clothing ID'] == clothing_id]
    rating = item_reviews['Recommended IND'].sum() * 2 - len(item_reviews)

    # Update and save the rating in the clothes DataFrame
    clothes_df.loc[clothes_df['Clothing ID'] == clothing_id, 'Rating'] = rating
    clothes_df.to_csv('data/clothes.csv', index=False)