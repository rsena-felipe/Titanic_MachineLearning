import my_functions
import pandas as pd

def make_predictions(input_file, model, index_column = "PassengerId", target_column = "Survived"):
    """
    Make predictions using a sklearn model and put the true label in the same df

    Arguments:
    input_file (str) -- Path to .csv that has the features of the previously trained model
    model -- Sklearn model previously trained
    index_column -- Index column of the dataset
    target_column -- Target column of the dataset 

    Returns:
    df -- A dataframe with three columns index, predictions and the true label
    """
    df = pd.read_csv(input_file).set_index(index_column)
    df.rename(columns = {"Survived":"True_Label"}, inplace = True)

    X, _ = my_functions.create_features_target(input_file, target_column = target_column, index_column = index_column)

    y_pred = pd.DataFrame(model.predict(X), index=df.index, columns = ["Prediction"])
    y_true = df["True_Label"]

    df = pd.concat([y_pred, y_true], axis=1)

    return df



def save_predictions(train_file, val_file, output_file, model, index_column = "PassengerId", target_column = "Survived"):
    """
    Make predictions on the train and validation data and saves it to a .csv

    Arguments:
    train_file (str) -- Path to .csv that has the features of the previously trained model
    val_file (str) -- Path to .csv that has the features of the validation data
    output_file (str) -- Path to .csv where the predictions are going to be saved
    index_column -- Index column of the dataset
    target_column -- Target column of the dataset 
    """

    # Make predictions for training
    df_train = make_predictions(train_file, model, index_column, target_column)
    df_train["split"] = "training"

    # Make predictions for validation
    df_val = make_predictions(val_file, model, index_column, target_column)
    df_val["split"] = "validation"
    df_val.to_csv(output_file)