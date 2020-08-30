import pandas as pd

def extract_ticket_prefix(df_column):
    """
    Extracts  ticket prefix, if no prefix returns X

    Arguments:
    df_column -- Pandas Dataframe Series (Ticket column of the titanic dataframe)

    Returns:
    Ticket_prefix -- List containing the ticket prefixes
    """

    Ticket_prefix = []
    for i in list(df_column):
        if not i.isdigit() :
            Ticket_prefix.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
        else:
            Ticket_prefix.append("X")
            
    return Ticket_prefix

def map_fsize(family_size):
    if family_size == 1:
        return "single"
    elif family_size == 2:
        return "small"
    elif family_size >= 3 and family_size <=4:
        return "medium"
    elif family_size >=5:
        return "large"     

def build_features_roche(input_file, output_file):
    """
    Change values of the column Sex male = 0 and female = 1.
    Change values of the column Embarked Southampton = 1, Cherbourg = 2 and Queenstown = 3.
    Creates a FamilySize column that is the sum of the SibSp and Parch column.
    
    Arguments:
    input_file (str) -- Path to the csv file already preprocessed.
    output_file (str) -- Path where the new csv of the new process data is going to be saved.

    Returns:
    A .csv of the processed dataframe in the specified location. 
    """

    dtypes = {"Pclass":"category", "Sex":"category", "Embarked":"category"} # Establishing the category data so we can create dummy variables later
    df = pd.read_csv(input_file, dtype = dtypes)

    df = pd.get_dummies(df) 
  
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1 # It sums the number of Sibling/Spouses (df["SibSp"]) and the number of Parents/Children (df["Parch"]) 

    df['IsAlone'] = df["FamilySize"].apply(lambda x: 0 if x == 1 else 1) # Check if someone is alone or not

    df.to_csv(output_file, index = False)       

def build_features_raul(input_file, output_file, split, sep=";"):
    """
    Formats the titanic dataset as follows:

    New Variables
    1. Is_Married: Boolean saying if someone is married or not (If Mrs in the name is married)
    2. Title: Title extracted from the name column. It is also group as follows:
        -Military/Politician/Clergy = Dr, Col, Major, Jonkheer, Capt, Sir, Don , Rev. 
        -Miss = Miss, Mrs, Ms, Mlle, Lady, Mme, the Countess, Dona.
        -Master remains the same, because it is a title of people under 26.
        -Mr remains Mr.        
    3. Fsize: SibSp + Parch + 1 (Size of the family)
    4. TypeF: Category based on family size (Fsize). It can be single, small, medium, large. 
    5. Ticket: Category based on Ticket index.

    Filling NA's
    1. Cabin(>600): Drop whole Cabin column, because it has more than 600 of NA's.
    2. Age(>150): Educated guess of median age of groupby of Sex and Pclass..
    3. Embarked(2): Drop NA's.
    
    Arguments:
    input_file (str) -- Path to the csv file already preprocessed.
    output_file (str) -- Path to the pickle where the new process data is going to be saved.
    split (str) -- The file is training or validation dataset.
    sep (str) -- How the csv is separated.

    Returns:
    A pickle of the processed dataframe in the output_file location. 
    """
    
    # Set data types (PassengerId as string because of index duplicates)
    dtype = {"PassengerId":"object"} 
    
    #, "Pclass":"category", "Sex":"category", "Embarked":"category"

    # Read csv
    df = pd.read_csv(input_file, sep=sep, dtype=dtype)

    # Create Index (Not duplicated)
    df["PassengerId"] = df["PassengerId"] + "-" + str(split)
    df.set_index("PassengerId", inplace = True)

    # Drop Cabin Column
    df.drop(["Cabin"], axis = 1, inplace=True) 

    # Extracts Title from Name column
    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0] 

    # Create Is_Married based on Title == Mrs
    df['Is_Married'] = df["Title"].apply(lambda x: 1 if x == 'Mrs' else 0)

    # Group titles in categories (Titles that signifies the same)
    df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Military/Politician/Clergy')
    df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss')

    # Drop Name Column
    df.drop(["Name"], axis = 1, inplace=True) # Drop Name Column 

    # Fill NA values of Age with median of Pclass, Sex
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    
    # Creates Family size descriptor from SibSp and Parch and this is going to be as a category
    df["Fsize"] = df["SibSp"] + df["Parch"] + 1 

    # Creates Type of Family column based on family size
    df["TypeF"] = df["Fsize"].apply(lambda fsize: map_fsize(fsize))

    # Creates Ticket extracting the prefix from the Ticket column
    df["Ticket"] = extract_ticket_prefix(df["Ticket"])

    # Drop resting Na values (Only 2 Na in Embarked column)
    df.dropna(inplace=True)

    # Format columns data type
    
    # Category (Important for the get dummies part)
    df["Ticket"] = df["Ticket"].astype("category")
    df["Title"] = df["Title"].astype("category")
    df["TypeF"] = df["TypeF"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    df["Pclass"] = df["Pclass"].astype("category")
    # Booleans
    df["Survived"] = df["Survived"].astype("bool")
    df["Is_Married"] = df["Is_Married"].astype("bool")
    df["Fsize"] = df["Fsize"].astype("bool")

    # Format Dataframe to get dummies (columns to be dummies need to be formated as category)
    df = pd.get_dummies(df)

    # Save as Pickle to conserve the data types
    df.to_pickle(output_file)

