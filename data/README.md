

## Content


```bash
├── features
│   ├── train_raul.pkl
│   ├── train_RF_original.csv
│   ├── val_raul.pkl
│   └── val_RF_original.csv
├── predictions
│   ├── ada_boosting.csv
│   ├── gradient_boosting.csv
│   ├── logreg_rfecv.csv
│   ├── random_forest.csv
│   ├── rf_raul_featuresoriginal.csv
│   ├── rf_original_featuresoriginal.csv
│   └── svc.csv
├── preprocess
│   ├── train_preprocess_original.csv
│   └── val_preprocess_original.csv
├── raw
│   ├── train.csv
│   └── val.csv

```

* features: train_raul.pkl and val_raul.pkl are serialized objects of the features extraction that I explained in "notebooks/1_EDA_FeatureEngineering.ipynb". train_RF_original.csv and val_RF_original.csv are the features extraction proposed in the original project.

* predictions: I save all predictions made by the models in this file. They contain the columns PasssengerId, Predictions, True_Label, and Split. The files: ada_boosting.csv, gradient_boosting.csv, logreg_rfecv.csv, random_forest.csv, and svc.csv are the results of models trained with the feature extraction made by myself and the training is explained in "notebooks/2_TrainModels.ipynb". The files: rf_raul_featuresoriginal.csv and rf_original_featuresoriginal.csv are the results of a random forest model trained with the feature extraction of the original code, the process of training is explained in the file: "notebooks/0_CleanCode.ipynb".

* preprocess: Contains the data that is preprocessed by the original code by you, this is explained in "notebooks/0_CleanCode.ipynb". I did not do a preprocess step, because I made preprocess and building features steps in a single function.

* raw: Is the data without any changes of the titanic task, split in training and validation sets. The contents of this raw data are explained below. 



## Raw Data Dictionary

<!-- TABLE_GENERATE_START -->

|   Variable    | Definition |Key |
| ------------- | ------------- | ------------- |
|     survival  |Survival   | Survival  0 = No, 1 = Yes     
survival           |
|     pclass    |Ticket class   |1 = 1st, 2 = 2nd, 3 = 3rd   |
|     sex       |Sex   |   |
|     Age       |Age in years   |   |
|     sibsp     | # of siblings / spouses aboard the Titanic  |   |
|     parch     |# of parents / children aboard the Titanic   |   |
|     ticket    | Ticket number  |   |
|     fare      |   Passenger fare   |   |
|     cabin     |   Cabin number   |   |
|     embarked  |   Port of Embarkation  | C = Cherbourg, Q = Queenstown, S = Southampton  |

<!-- TABLE_GENERATE_END -->

## Variables Notes
 
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children traveled only with a nanny, therefore parch=0 for them.
