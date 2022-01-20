# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [Project Descriptions](#project-descriptions)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Authors](#author)

Project Motivation

In this project, I applied my data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. I have created a machine learning pipeline to categorize real messages that were sent during disaster events so that the messages could be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
File Descriptions


Project Description

This Projects consits of three different parts which will be descripted below:

1. ETL Pipeline
Python scripts, which which wirtes da data cleaning pipeline that:
    Loads two datasets (messages and categories)
    Merges and then cleans both datasets
    and stores the result in a SQLite database (DiasterResponse.db)


2. ML Pipeline

A Python script, train_classifier.py, writes a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file


3. Flask Web App

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.