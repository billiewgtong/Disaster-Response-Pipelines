# Disaster Response Pipelines

In this project, I analyze disaster data from Figure Eight by building a model that classifies disaster messages. Specifically the follow are built/advanced:

    I. ETL Pipeline
      - Loads the messages and categories datasets
      - Merges the two datasets
      - Cleans the data
      - Stores it in a SQLite database

    II. ML Pipeline
      - Loads data from the SQLite database
      - Splits the dataset into training and test sets
      - Builds a text processing and machine learning pipeline
      - Trains and tunes a model using GridSearchCV
      - Outputs results on the test set
      - Exports the final model as a pickle file

    III. Flask Web App
      - Modify file paths for database and model as needed
      - Add data visualizations using Plotly in the web app

## Files Used

     * 'ETL Pipeline Preparation.ipynb': Consists of ETL pipeline work. 
     * 'ML Pipeline Preparation.ipynb': Consists of ML pipeline work.
     * 'train_classifier.py': Consists of definitions that automate the ETL pipeline work.
     * 'process_data.py': Consists of definitions that automate the ML pipeline work.
     * 'run.py': Consists of a script in takes the ETL and ML work to create a web app.

## Data
```
The datasets used in this analysis are:

  I. Disaster Messages retrieved from Figure 8 via Udacity.
    
  II. Disaster Categories retrieved from Figure 8 via Udacity.
```

## Languages, Programs and Libraries

    * Python 3
    * Anaconda
    * Pandas, Numpy, Sklearn, Re, Sqlalchemy, Pickle, and NLTK


## Authors

    * *Billie Tong* - *Analytical work and writing*

