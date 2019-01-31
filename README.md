# DisasterResponse
---

### Summary
The object of this project was to build a text classifier that identifies emergency services needs during/after a natural disaster. The whole process, from consuming data to saving a working model, is streamlined using two pipelines; one for ETL (extract, transform, load), and the other for building a machine learning model. The project also contains a simple web app that takes a text input from a user and classifies into appropriate categories using the model.

### Software requirements
This project is created using python 3.7, scikit-learn 0.20.1, nltk 3.4, and flask 1.0.2.

### Installation Guide
To test this model:

1) clone this repository.

2) To pre-process data, run process_data.py. The process_data.py script takes three arguments; a csv file containing messages, a csv file containing disaster categories, and the file path where the processed data should be stored.
Example: python process_data.py messages.csv categories.csv DisasterResponse.db

3) To build and train the classifier, run train_classifier.py. The train_classifier.py takes two arguments; a path of the cleaned data, and a path where the model is stored.
Example: python train_classifier.py DisasterResponse.db classifier.pickle

4) To run a web app locally, go to the app folder and execute run.py file (python run.py). Then open a web browser and go to 'http://0.0.0.0:3001/'

### Data Source
The dataset was provided by Figure Eight Inc. through Udacity.

### License
This project is shared under the MIT license.
