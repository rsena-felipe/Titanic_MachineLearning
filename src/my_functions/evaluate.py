import my_functions
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt

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

    df = df.replace({0: 'die', 1: 'live'})

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
    
    df = df_train.append(df_val)
    df.to_csv(output_file)

# Metrics Plots

def plot_confusion_matrix(input_file, split):    
    
    df = pd.read_csv(input_file)
    df = df[df['split'] == split]

    labels = ['live', 'die']
    
    cm = confusion_matrix(df.True_Label, df.Prediction, normalize = 'true')
    cm = cm*100 # Multiply by 100 so te percentage is 99.7% instead of 0.997%
    cm = pd.DataFrame(cm, index=labels, columns=labels)        

    # Customize heatmap (Confusion Matrix)
    sns.set(font_scale = 1.5)
    ax = sns.heatmap(cm, cmap='BuGn', annot=True, annot_kws={"size": 18,"weight":"bold"}, cbar=False, fmt='.3g')
    for t in ax.texts: t.set_text(t.get_text() + " %") # Put percentage in confusion matrix
    plt.xlabel('Predicted',weight='bold')
    plt.ylabel('Known',weight='bold')
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0)

def plot_confusion_matrices(input_file):
    
    plt.rcParams['figure.figsize'] = [18, 5]
    plt.subplot(1,2,1)
    plot_confusion_matrix(input_file, "training")
    plt.title("Matrix Training")
    plt.subplot(1,2,2)
    plot_confusion_matrix(input_file, "validation")
    plt.title("Matrix Validation")

def plot_classification_report(input_file, split):

    df = pd.read_csv(input_file)
    df = df[df['split'] == split]

    report = pd.DataFrame(classification_report(df.True_Label, df.Prediction, digits=3, output_dict=True)).transpose()
    report = report.loc[:, ["precision", "recall", "f1-score"]].drop('accuracy')
    report = report*100 # Multiply by 100 so te percentage is 99.7% instead of 0.997%
    
    # Customize heatmap (Classification Report)
    sns.set(font_scale = 1.3)
    rdgn = sns.diverging_palette(h_neg=10, h_pos=130, s=80, l=62, sep=3, as_cmap=True)
    
    ax=sns.heatmap(report, cmap=rdgn, annot=True, annot_kws={"size": 14}, cbar=True, fmt='.3g', cbar_kws={'label':'%'}, center=90, vmin=0, vmax=100)
    ax.xaxis.tick_top()
    for t in ax.texts: t.set_text(t.get_text() + " %") #Put percentage in confusion matrix

def plot_classification_reports(input_file):

    plt.rcParams['figure.figsize'] = [18, 5]
    plt.subplot(1,2,1)
    plot_classification_report(input_file, split="training")
    plt.title("Training")
    plt.subplot(1,2,2)
    plot_classification_report(input_file, split="validation")
    plt.title("Validation")

def print_accuracies(input_file):

    df = pd.read_csv(input_file)
    splits = ["training", "validation"]

    for split in splits:
        df_intermediate = df[df['split'] == split]
        print("The " + str(split) + " accuracy is: ", round(accuracy_score(df_intermediate.True_Label, df_intermediate.Prediction)*100,2), "%")
        print("The " + str(split) + " balanced accuracy is: ", round(balanced_accuracy_score(df_intermediate.True_Label, df_intermediate.Prediction)*100,2), "%")
        print()

    # # Print accuracy Training
    # df = pd.read_csv(input_file)
    # df_train = df[df['split'] == 'training']

    # print("The training accuracy is: ", round(accuracy_score(df_train.True_Label, df_train.Prediction)*100,2), "%")
    # print("The training balanced accuracy is: ", round(balanced_accuracy_score(df_train.True_Label, df_train.Prediction)*100,2), "%")

    # print() 

    # # Print accuracy Validation
    # df = pd.read_csv(input_file)
    # df_validation = df[df['split'] == 'validation']

    # print("The validation accuracy is: ", round(accuracy_score(df_validation.True_Label, df_validation.Prediction)*100,2), "%")
    # print("The validation balanced accuracy is: ", round(balanced_accuracy_score(df_validation.True_Label, df_validation.Prediction)*100,2), "%")