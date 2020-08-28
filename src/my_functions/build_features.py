import pandas as pd

def build_features_roche(input_file, output_file):
    """
    Change values of the column Sex male = 0 and female = 1.
    Change values of the column Embarked Southampton = 1, Cherbourg = 2 and Queenstown = 3.
    Creates a FamilySize column that is the sum of the SibSp and Parch column.
    
    Arguments:
    input_file -- String Path to the csv file already preprocessed.
    output_file -- String Path where the new csv of the new process data is going to be saved.

    Returns:
    A .csv of the processed dataframe in the specified location. 
    """

    dtypes = {"Pclass":"category", "Sex":"category", "Embarked":"category"} # Establishing the category data so we can create dummy variables later
    df = pd.read_csv(input_file, dtype = dtypes)

    df = pd.get_dummies(df) 
  
    df["FamilySize"] = df["SibSp"] + df["Parch"] # It sums the number of Sibling / Spouses (df["SibSp"]) and the number of Parents / Children (df["Parch"]) 

    df['IsAlone'] = df["FamilySize"].apply(lambda x: 0 if x == 0 else 1) # Check if he is alone or not

    df.to_csv(output_file, index = False)    