import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Takes a file paths of messages and categories csv files and returns a
    dataframe containing messages and its corresponding categories.

    Parameters:
    messages_filepath (string): a file path of a messages csv file
    categories_filepath (string): a file path of a categories csv file

    Returns:
    df (pandas dataframe): a dataframe containing messages and their
    corresponding categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.join(categories, rsuffix="_categories")

    return df


def clean_data(df):
    """
    Takes a dataframe, one-hot-encodes the categories, removes unnecessary
    columns, and removes duplicates from the dataframe.

    Parameters:
    df (pandas dataframe): a dataframe containing messages and their
    corresponding categories

    Returns:
    df (pandas dataframe): a cleaned dataframe to be saved to sql database
    """
    # One-hot-encode the categories
    categories = df["categories"].str.split(";", expand=True) # Expand the category column to individual category columns

    # Extract the values from the category strings and enter the values into corresponding columns
    category_colnames = lambda x: [str(y)[:-2] for y in x]
    row = categories.iloc[0]
    categories.columns = category_colnames(row)

    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
        categories[column] = categories[column].apply(lambda x: int(x))

    # Clean up and add the categories df
    df.drop(["id_categories", "categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(keep=False, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Takes a dataframe and a database filename, and saves the dataframe in a
    database

    Parameters:
    df (pandas dataframe): a cleaned dataframe containing messages and one-hot-
    encoded categories to be saved
    database_filename (string): a name of the sql database

    Returns:
    None
    """
    # Save the df to the SQL db
    db_path_name = 'sqlite:///' + database_filename
    engine = create_engine(db_path_name)
    df.to_sql('Messages_Categories', engine, index=False)


def main():
    """
    Takes file paths of messaves and categories csv files and the file path of
    a sql database to be saved

    Parameters:
    message_filepath (string): a file path of a messages csv file
    categories_filepath (string): a file path of a categories csv file
    database_filepath (string): a file path of a database

    Returns:
    None, but saves a sql db file at a specified path
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
