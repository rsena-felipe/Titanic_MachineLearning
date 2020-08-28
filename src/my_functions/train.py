import pandas as pd


def create_features_target(input_file, target_column, index_column):
    """
    Function to divide a dataframe in features (X) and target (y)

    Arguments:
    input_file -- String Path to the csv file of the dataset (The csv separator should be ",")
    target_column -- String name of the target column (y), all the columns that are not the target column are going to be features

    Returns:
    X -- A numpy.ndarray of features
    y -- A numpy.ndarray of the target
    """

    df = pd.read_csv(input_file)

    df = df.loc[:, df.columns != index_column]

    X = df.loc[:, df.columns != target_column].values
    y = df[target_column].values

    return X, y

