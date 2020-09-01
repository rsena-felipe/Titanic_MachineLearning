

## Content


```bash
.
├── deep_learning
│   └── history.csv
├── features_raul
│   ├── ada_boosting.pkl
│   ├── gradient_boosting.pkl
│   ├── logreg_rfecv.pkl
│   ├── random_forest.pkl
│   └── svc.pkl
├── features_roche
│   ├── rf_raul.pkl
│   └── rf_roche.pkl
├── models_api
│   ├── svc_model_columns.pkl
│   └── svc.pkl
```


<!-- TABLE_GENERATE_START -->

| Folder  | Content |
| ------------- | ------------- |
| Deep Learning  | history.csv is the history of the trained model via DL. I did not save the model, because it was not good.  |
| Features Raul  | All models trained by my features extraction  |
| Features Roche  | Random Forest trained with original features extraction  |
| Models API  | Suppport Vector Classifier Model trained with the "prediction_api/model.py" sames hyperparameters as the svc in "models/features_raul/svc.pkl"  this is the file svc.pkl. I serialized all columns from training in svc_models_columns.pkl as a solution to the less than expected number of columns.|

<!-- TABLE_GENERATE_END -->