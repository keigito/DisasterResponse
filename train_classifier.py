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
nltk.download(['punkt', 'wordnet'])

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

if __name__ == "__main__":

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('Messages_Categories', engine)

    X = df.loc[:, "message"]
    Y = df.iloc[:, 4:40]

    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs=4))
    ])

    parameters = {
#    'vectorizer__ngram_range': ((1, 1), (1, 2)),
#    'vectorizer__max_df': (0.5, 1.0),
    'vectorizer__max_features': (5000, 10000), # 397
    'tfidf__use_idf': (True, False), # 489 sec
#    'clf__estimator__n_estimators': [100, 200], #
    'clf__estimator__min_samples_split': [3, 4] # 444 sec
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    cv_results = cv.fit(X_train, y_train)

    best_model = cv_results.best_estimator_

    pickle.dump(best_model, open("final_model.pickle", "wb"))
