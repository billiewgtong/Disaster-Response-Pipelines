import sys
import numpy as pd
import pandas as pd
import pickle
import matplotlib.pyplot as py
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

def load_data(database_filepath):
    
    '''
    INPUT
    database_filepath - string of database filepath
    
    OUTPUT
    X and Y - dataframes for machine learning processing
    category_names - names of categories
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_details', engine)
    X = df['message']
    Y = df.drop(columns = ['id', 'message' , 'original' ,'genre'])
    
    # Get names of all categories
    category_names = Y.columns.tolist()
    
    return X, Y, category_names    


def tokenize(text):
    
    '''
    INPUT
    text - string of text for processing
    
    OUTPUT
    new_tokens - list of processed tokens from the input string of text
    '''
    
    reg_exp_non_anum = r"[^a-zA-Z0-9]"
    
    #Convert all text to lower case
    text = text.lower()
    
    #Remove all non-alphanumeric characterics with a space to prevent meshed words
    text = re.sub(reg_exp_non_anum, " ", text)
    
    #Tokenize text
    tokens = word_tokenize(text)
    
    #Instantiate Stemmer and Lemmitizer 
    stemmer = PorterStemmer()
    lemmitizer = WordNetLemmatizer()
    
    #Convert raw tokens into lemmitized and stemmed forms w/ no white spaces
    new_tokens = []
    
    for token in tokens:
        lem_token_noun = lemmitizer.lemmatize(token)
        lem_token_verb = lemmitizer.lemmatize(lem_token_noun, pos = 'v')
        stem_lem_token = stemmer.stem(lem_token_verb)
        new_token = stem_lem_token.strip()
        new_tokens.append(new_token)
        
    return new_tokens


def build_model(version = 1):
    
    '''
    INPUT
    version - default version: 1. Determines which pipeline is used. Version 1: tfidfVectorizer 
    and Multioutput Randomforest classifier. Version 2: HashVectorizer and Multioutput Adaboost
    classifier.
    
    OUTPUT
    cv_model - a model that utilizes a prespecified pipeline for vectorization and classification
    in combination with gridsearch to find best specified parameters for predictions
    '''
    if version == 1:
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer= tokenize)),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 30)))
        ])
        
        parameters = {
         'clf__estimator__n_estimators': (5 ,10)
        }
    
        cv_model = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
        
        
    elif version == 2:        
        pipeline = Pipeline([
        ('hashing', HashingVectorizer(tokenizer= tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state = 30)))])

        parameters = {
            'clf__estimator__learning_rate': (0.6,1.0)
            }
    
        cv_model = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
        
    else:
        print("Enter 1 or 2")
               
    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    INPUT
    model - a model that utilizes a prespecified pipeline for vectorization and classification
    in combination with gridsearch to find best specified parameters for predictions
    X_test - test input data obtained from train test split
    Y_test - test output data obtained from train test split
    category_names - the names of the Y (i.e. Output) categories
    
    OUTPUT
    Tables containing stats on how well the model predicts the test data
    '''
    Y_pred = model.predict(X_test)
    
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    
    for i in range(36):
        
        print('Category: {}'.format(category_names[i].upper()), "\n\n",
             classification_report(Y_test.iloc[:,i], Y_pred_df.iloc[:,i]))


def save_model(model, model_filepath):
    
    '''
    INPUT
    model - the name in which the file would be saved as
    model_filepath - the filepath in which the file would be saved
    
    OUTPUT
    File saved with desired name and filepath
    '''
    
    filename = model
    pickle.dump(filename, open(model_filepath, 'wb'))


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