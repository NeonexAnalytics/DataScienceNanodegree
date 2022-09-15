import sys
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')



import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    print(database_filepath)
    # load data from database
    engine = create_engine("sqlite:///" + database_filepath, pool_pre_ping=True)
    df = pd.read_sql_table("messages", con = engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names

def tokenize(text):
     # Tokenize the string text and initiate the lemmatizer
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    # detect urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #normalize, remove punctuation and then tokenize
    words = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    #Remove stopwords (english)
    tokens = [word for word in words if word not in stopwords.words("english")]
    
    #Lemmatize
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    pipeline_adaboost = Pipeline([
    ("vect", CountVectorizer(tokenizer=tokenize)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(AdaBoostClassifier()))
        ])
    parameters = {"vect__max_features":[200, 500, 1000]}
    
    cv = GridSearchCV(pipeline_adaboost, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # Predict the given X_test and create the report based on the Y_pred
    scores = []
    counter = 0
    y_pred = model.predict(X_test)
    for feature in Y_test:
        print("Feature - {}: {}".format(counter+1, feature))
        print(classification_report(Y_test[feature], y_pred[:,counter]))
        acc = accuracy_score(Y_test.iloc[:,counter], y_pred[:,counter])
        scores.append(acc)
        counter = counter+1
    print("Total Accuracy: {}".format(np.mean(scores)))
    return np.mean(scores)
           


def save_model(model, model_filepath):
    # Save the model based on model_filepath given
    pkl_filename = '{}'.format(model_filepath)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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