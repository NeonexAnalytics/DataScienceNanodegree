# Disaster Response Pipeline with Figure Eight

### Project Motivation
This Project is part of the Udacity Data Science Nanodegree Program

The project provides a web app where you can input a text message and reive a classification in 36 different categories and provided overal results in visualization representation.

### Project Description
The project consits of three parts:
- ETL Pipeline: process_data.py file with python code to create an ETL pipeline.
- ML Pipeline: train_classifier.py file with python code to create an ML pipeline.
- Web APP:


### Needed packages:
```
- pandas
- re
- json
- sklearn
- nltk
- sqlalchemy
- sqlite3
- pickle
- Flask
- plotly
```


### Instructions to run the Web App:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disasterPipeline_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors and Achknowledgements
Thanks to Udacity for providing the basic code of the WebAPP and both ETL and ML pipeline

