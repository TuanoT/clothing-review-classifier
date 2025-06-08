"""
The module creates a CSV file containing unique clothing items from the review dataset.
It needs to be run whenever new review data is added.
"""

import pandas as pd

def generate_csv():

    # Load the review data
    df = pd.read_csv('data/reviews.csv')

    # Remove duplicate reviews for the same clothing item
    df_clothes = df.drop_duplicates(subset='Clothing ID', keep='first')
    df_clothes = df_clothes.drop(columns=['Age', 'Title', 'Review Text', 'Rating', 'Recommended IND', 'Positive Feedback Count'])

    # Create a new column for search text
    df_clothes['Search Text'] = (
        df_clothes['Clothes Title'] + ' ' +
        df_clothes['Class Name'] + ' ' +
        df_clothes['Department Name'] + ' ' +
        df_clothes['Division Name'] + ' ' +
        df_clothes['Clothing ID'].astype(str)
    ).str.lower()

    # Create a new column for rating
    df_clothes['Rating'] = 0

    # Create .csv with clothing items only
    df_clothes.to_csv('data/clothes.csv', index=False)
    print("\n'data/clothes.csv' created successfully.")


if __name__ == "__main__":
    generate_csv()