# Titanic - Project 

This project uses machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

## Project Structure

The project is structure as follows: 

```bash
.
├──configurations.yml
│
├── data
│
├── models
│
├── notebooks
│   ├── 0_CleanCode.ipynb
│   ├── 1_EDA_FeatureEngineering.ipynb
│   ├── 2_TrainModels.ipynb
│   ├── 3_EvaluateModels.ipynb
│   
├── prediction_api
│   ├── api.py
│   ├── model.py
│   ├── proof_predicting_api.pdf
│
└── src
     ├── my_functions
     
```        

## Content

* The virtual environment to reproduce the results is in the configurations.yml.

* data contains the raw data, preprocess data, features data, and the predictions made for all models.

* models contain all models trained in pkl format.

* notebooks contains Exploratory Data Analysis, Feature Engineering, Training and Evaluation of Models.

* prediction_api contains code to share an SVC model via API. The api.py is the script to raise the port, model.py is a script to train a Support Vector Classifier (I could also use some of the already trained models, but I prefer to create a new model from zero)

* src contains the package my_functions. A package of functions that I will reuse in scripts, notebooks, etc. With this, I apply the concept of DRY (Do not repeat yourself). This package is in "src/my_functions". Almost all tasks use functions from this package. 

Feel free to contact me at raulfelipe.sena@gmail.com in case of any doubts.
