# Titanic - Project 

## Project Structure

The project is structure as follows: 

```bash
.
├──configurations.yml
│
├── data
│
├── docs
│
├── models
│
├── notebooks
│   ├── 0_Task3_RandomForest_roche.ipynb
│   ├── 1_Task4_FeatureEngineering_EDA.ipynb
│   ├── 2_Task5_Models.ipynb
│   ├── 3_Task6_EvaluateModels.ipynb
│   
├── prediction_api
│   ├── api.py
│   ├── model.py
│   ├── proof_predicting_api.pdf
│
└── src
     ├── my_functions
     │
     └── tests
```        

## Tasks Localization
I create a package of functions that I will reuse in scripts, notebooks, etc. With this, I apply the concept of DRY (Do not repeat yourself). This package is in "src/my_functions". Almost all tasks use functions from this package.

* Task 2 - a sense of humor is in docs folder.

* Task 3, 4, 5, and 6 (Good Practices, Feature Engineering, Models, and Measures) are in the folder notebooks.

* Task 7 - docker is the file configurations.yml

* Task 8 - tests are in "src/tests" Also I attached proof of running tests on the 

* Task 9 - prediction API is in the prediction_api folder. Their api.py is the script to raise the port, model.py is a script to train a support vector classifier (I could also use some of the already trained models, but I prefer to create a new model from zero)

## Folders Content

* data contains the raw data, preprocess data, features data, and the predictions made for all models.
* docs contain the Joke of Task2, enjoy =).
* models contain all models trained in pkl format.
* notebooks contain the tasks 3 to 6.
* prediction_api contains code to share an SVC model via API
* src contains the package my_functions and the unit tests.

Also, feel free to contact me at raulfelipe.sena@gmail.com in case of any doubts.
