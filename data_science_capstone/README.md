# Capstone Project: Sparkify

### Project Motivation
This Project is part of the Udacity Data Science Nanodegree Program

The imaginary start-up company Sparkify offers music streaming service to users in the USA. Many users stream their favourite songs to our service every day either using the free tier that places advertisements between the songs, or using the premium subscription model, where they stream the music for free but pay a monthly flat rate. Users can upgrade, downgrade, or cancel their service at any time. So, it is crucial to make sure the users love the service. Every time a user interacts with the service such as playing songs, logging out, liking a song with a thumps-up, hearing an ad, or downgrading their service, it generates data. All this data contains key insides for keeping the users happy and helping Sparkify’s business thrive. It is our job on the data team to predict which users are at risk to churn either downgrade from premium to free tier or cancelling their service altogether. If we can accurately identify these users before they leave, Sparkify can offer them discounts and incentives, potentially saving the business millions in revenue.

Sparkify keeps a Log file with 18 fields for every user interaction (e.g., userId, name of song played, length of song played, name of artist). Soon the data volume of the log file has exceeded the available memory space on standard desktop computers, and the company has opted for using the distributed file system Apache Spark™. Udacity™ provides the full dataset with 12GB on AWS™ S3, and you can run a Spark cluster on the cloud using AWS  to analyse the large amount of data.



### Project Description
- Load data into Spark
- Explore and Clean Data
- Create Features to predict churn event
- Build Model
- Predict churn


In the beginning of this project we use Spark SQL and Spark Dataframes to analyse a small subset of the Sparkify user data on local machine to test our model. We use PySpark to predict churn events of Sparkify users

### Needed packages:
```
- PySpark Machine Learning Library
- Pysparl SQL Module
- Pandas
- Numpy
- matplotlib
- seaborn
```




### Licensing, Authors and Achknowledgements
Thanks to Udacity for providing the data and Amazon for providing free cloud service essentials to performe this project

