# Import dependencies
import sys
sys.path.append("../src/")
import my_functions
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import joblib

# Load the dataset in a dataframe object 
features_train_file = "../data/features/train_raul.pkl"

# Extract Features and Target
X_train, y_train = my_functions.create_features_target(features_train_file, target_column="Survived", index_column="PassengerId", format_type="pickle")

# Support Vector classifier
svc = SVC()
svc.fit(X_train, y_train)

# Save your model
joblib.dump(svc, '../models/models_api/svc.pkl')
print("Model dumped!")

# Load the model that you just saved
svc = joblib.load('../models/models_api/svc.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, '../models/models_api/svc_model_columns.pkl')
print("Models columns dumped!")