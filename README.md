# Disaster Response Pipeline Project

## Project Motivation

In this project, I apply skills I learned in Data Engineering Section to analyze disaster data from Figure Eight to build a Classifier model to classify disaster messages.

## Process Flow
  - Clean and Store the Data (process_data.py)
    - Read messages and categories data file
    - Combined the data files, perform cleansing and store the cleaned data into a sqlite database
  
  - Create, Train and Store Classifier (train_classifier.py)
    - The train_classifier.py script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
    - The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.
    - The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.
  
  - Web app
    - the home page has visualization showing classification of the messages
    - The user can enter a message and view how the model classifies the message based on the trained model

To learn more about TF-IDF please follow the below links:
[http://www.tfidf.com/](TF-IDF - A Single Page Tutorial)
[https://www.elephate.com/blog/what-is-tf-idf/](What is TF-IDF)
[https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3](Medium Post of TF-IDF)

## File Description
    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # Train ML model           
    └── README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![ScreenShot](homepage.png)

## Examples

  - Message 1 : We need help. no food and water.

![Help](help.png)

  - Message 2 : accident on highway. we are injured. need help.
![Medical](medical.png)
