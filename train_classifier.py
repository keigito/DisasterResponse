import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
import nltk


def load_data(database_filepath):
    """
    Takes a database file path and returns features and target dataframes, as
    well as the names of categories

    Parameters:
    database_filepath (string): a file path of the database

    Returns:
    X (pandas dataframe): a dataframe containing text messages (a feature)
    Y (pandas dataframe): a dataframe containing one-hot-encoded categories
    category_names (list(string)): a list containing names of categories
    """
    db_path_name = 'sqlite:///' + database_filepath
    engine = create_engine(db_path_name)
    df = pd.read_sql_table('Messages_Categories', engine)

    X = df.loc[:, "message"]
    Y = df.iloc[:, 4:40]

    category_names = list(df.columns)
    drop_names = ["id", "message", "original", "genre"]
    for name in drop_names:
        category_names.remove(name)

    return X, Y, category_names


def tokenize(text):
    """
    Takes a text, normalizes, tokenizes, removes stop words, lemmatizes, and
    returns a list of tokens

    Parameters:
    text (string): an input text

    Returns:
    clean_tokens (list(string)): a list of cleaned tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
    Builds and returns a machine learning pipleline

    Parameters:
    None

    Returns:
    pipeline (Pipeline): a pipeline to build a classifier
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
    #     (), # Feature engineering (word2vec/GloVe)
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs=-1))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes model, X_test, Y_test, and category_names, and prints out a
    performance metrics for each categories

    Parameters:
    model (sklearn model): a classifier to be tested
    X_test (Pandas Series): A series containing texts to be classified
    Y_test (Pandas Series): A series containing categories to be used for
    validation
    category_names (list(string)): a list of category names

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    y_actu = Y_test.values

    results_dict = {}
    for i in range(1, 37):
        predicted = "pred_" + str(i)
        actual = "actu_" + str(i)
        pred_values = []
        actu_values = []
        for ii in range(len(y_pred)):

            pred_values.append(int(y_pred[ii][i-1]))
            actu_values.append(int(y_actu[ii][i-1]))

        results_dict[predicted] = pred_values
        results_dict[actual] = actu_values

    for i in range(1, 37):
        pred = results_dict['pred_' + str(i)]
        actu = results_dict['actu_' + str(i)]

        print("\n### " + category_names[i-1] + " ###\n")
        print(classification_report(pred, actu))



def save_model(model, model_filepath):
    """
    Takes a model and the path to save the model

    Parameters:
    model (sklearn model): a classifier to be saved
    model_filepath (string): a file path where the model will be saved

    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Takes database_filepath and the model_filepath, and saves the trained
    classifier model at the specified path

    Parameters:
    database_filepath (string): a file path of the database to use
    model_filepath (string): a file path of the model to be saved

    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        nltk.download(['punkt', 'wordnet'])
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
