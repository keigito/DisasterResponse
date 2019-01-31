import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import numpy as np


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../DisasterResponse.db')
df = pd.read_sql_table('Messages_Categories', engine)

# load model
model = joblib.load("../classifier.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    aid_names = ["medical_help", "medical_products", "search_and_rescue", "security", "military", "child_alone", "water", "food", "shelter", "clothing", "money", "missing_people", "refugees", "death", "other_aid"]
    aid_counts = []
    for name in aid_names:
        aid_counts.append(df[(df[name] == 1)][name].count())
    aid_counts = np.array(aid_counts)

    infrastructure_names = ["transport", "buildings", "electricity", "tools", "hospitals", "shops", "aid_centers", "other_infrastructure"]
    infrastructure_counts = []
    for name in infrastructure_names:
        infrastructure_counts.append(df[(df[name] == 1)][name].count())
    infrastructure_counts = np.array(infrastructure_counts)

    weather_names = ["floods", "storm", "fire", "earthquake", "cold", "other_weather"]
    weather_counts = []
    for name in weather_names:
        weather_counts.append(df[(df[name] == 1)][name].count())
    weather_counts = np.array(weather_counts)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=aid_names,
                    y=aid_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Aid-Related Messages by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=infrastructure_names,
                    y=infrastructure_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Infrastructure-Related Messages by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=weather_names,
                    y=weather_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Weather-Related Messages by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
