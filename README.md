# Titanic - Project 

## Project Structure

The project is structure as follows: 

```bash
.
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
I create a package of functions that I will reuse in scripts, notebooks, etc. With this I apply the concept of DRY (Do not repeat yourself). This package is in "src/my_functions". Almost all tasks uses functions from this package.

* Task 2 is in docs folder.

* Task 3, 4, 5 and 6 are in the folder notebooks.

* Task 7 is

* Task 8 is in "src/tests"

* Task 9 is in prediction_api folder. There api.py is the script to raise the port, model.py is a script to train a support vector classifier (I could also use some of the already trained models, but I prefer to create a new model from zero)

## Folders Content

* data contains the raw data, preprocess data, features data and the predictions made for all models.
* docs contains the Joke of Task2, enjoy =).
* models contains all models trained in pkl format.
* notebooks contains the tasks 3 to 6.
* prediction_api contains code to share an SVC model via API
* src contains the package my_functions and the unit tests.

Also, feel free to contact me at: raulfelipe.sena@gmail.com in case of any doubts.
