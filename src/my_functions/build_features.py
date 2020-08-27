import pandas as pd

def build_features_roche(input_file, output_file):
    """
    Change values of the column Sex with male = 0 and female = 1.
    Change values of the column Embarked with Southampton = 1, Cherbourg = 2 and Queenstown = 3.
    Creates a FamilySize column that is the sum of the SibSp and Parch column.
    
    Arguments:
    input_file -- String Path to the csv file already preprocessed.
    output_file -- String Path where the new csv of the new process data is going to be saved.

    Returns:
    A .csv of the processed dataframe in the specified location. 
    """
    df = pd.read_csv(input_file)

    df["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    df["Embarked"].replace({"Southampton": 1, "Cherbourg": 2, "Queenstown": 3}, inplace=True)
  
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1 # It sums the number of Sibling / Spouses (df["SibSp"]) and the number of Parents / Children (df["Parch"]) 

    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    return df.to_csv(output_file, index = False)    