import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

SUCCESS_STR = "+"
FAILURE_STR = "-"
def preprocess_data(df):
    df.drop("salary", axis=1, inplace=True)
    df.drop("salary_currency", axis=1, inplace=True)

    # check if there are any missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("No missing values found")
    else:
        print(f"Missing values found: {missing_values}")

        # remove rows with three or more missing values if the
        data_cleaned = df.dropna(thresh=3).copy()

        removed_rows = df.shape[0] - data_cleaned.shape[0]
        removed_rows_percent = (removed_rows / df.shape[0]) * 100
        print(f"Removed rows:{removed_rows}, {removed_rows_percent}%")

        missing_values_count = data_cleaned.isnull().sum().sum()
        if missing_values_count == 0:
            print("No missing values found")
        else:
            print("Filling in missing values...")
            # fill in missing values with the mean of the column and most frequent value
            for column in data_cleaned.columns:
                if data_cleaned[column].dtype == "int64":
                    data_cleaned.loc[:, column] = data_cleaned[column].fillna(data_cleaned[column].mean())
                else:
                    data_cleaned.loc[:, column] = data_cleaned[column].fillna(data_cleaned[column].mode()[0])

            replaced_values_count = missing_values_count - data_cleaned.isnull().sum().sum()
            replaced_values_percent = (replaced_values_count / missing_values_count) * 100
            log_replaced_values = f"Replaced missing values percent: {replaced_values_percent}%"

            print(f"Replaced values:{replaced_values_count}, {log_replaced_values}")
        df = data_cleaned.copy()

    # convert to appropriate data types
    print("Converting data types...")
    success = True
    label_encoders = {}
    scalers = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder().fit(df[column])
            label_encoders[column] = le
            df.loc[:, column] = le.fit_transform(df[column])
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "int64":
            scaler = MinMaxScaler().fit(df[column].values.reshape(-1, 1))
            scalers[column] = scaler
            df.loc[:, column] = scaler.transform(df[column].values.reshape(-1, 1)).astype("float")
        else:
            success = False
            print(f"Column {column} not converted as it is of type {df[column].dtype}")
    if success:
        print(f"{SUCCESS_STR}Data types converted successfully")
    else:
        print(f"{FAILURE_STR}Failed to convert some data types")

    # reset the index
    df.reset_index(drop=True)

    return df