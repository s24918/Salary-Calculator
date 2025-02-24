import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import pickle

columns = ['job_title', 'experience_level', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']

SUCCESS_STR = "+"
FAILURE_STR = "-"
def preprocess_data(df, train=False):
    if 'salary' in df.columns:
        df.drop("salary", axis=1, inplace=True)
    if 'salary_currency' in df.columns:
        df.drop("salary_currency", axis=1, inplace=True)
    if 'employment_type' in df.columns:
        df.drop("employment_type", axis=1, inplace=True)
    
    df['job_title'] = df['job_title'].replace('Machine Learning Engineer', 'ML Engineer')

    # replace rare values with "other"
    if train:
        for col in df.columns:
            if df[col].dtype == "object":
                value_counts = df[col].value_counts()
                df.loc[~df[col].isin(value_counts[value_counts > 10].index), col] = "other"


    # check if there are any missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("No missing values found")
    else:
        print(f"Missing values found: {missing_values}")

        # remove rows with three or more missing values
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
    if train:
        label_encoders = {}
        scalers = {}
    else:
        with open("model/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open("model/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
    for column in df.columns:
        if df[column].dtype == "object":
            if train:
                le = LabelEncoder().fit(df[column])
                label_encoders[column] = le
                df[column] = le.fit_transform(df[column])
            else:
                le = label_encoders[column]
                try:
                    df[column] = le.transform(df[column])
                except ValueError as e:
                    raise ValueError(f"Failed to convert column {column} due to {e}")
            df[column] = df[column].astype("int32")
        elif str(df[column].dtype).startswith("int") or str(df[column].dtype).startswith("float"):
            if train:
                scaler = MinMaxScaler().fit(df[column].values.reshape(-1, 1))
                scalers[column] = scaler
            else:
                scaler = scalers[column]
            try:
                # Convert to float64 before scaling
                df[column] = df[column].astype("float64")
                df[column] = scaler.transform(df[column].values.reshape(-1, 1)).ravel()
            except Exception as e:
                raise ValueError(f"Failed to convert column {column} due to {e}")
        else:
            success = False
            print(f"Column {column} not converted as it is of type {df[column].dtype}")
    if success:
        print(f"{SUCCESS_STR}Data types converted successfully")
    else:
        print(f"{FAILURE_STR}Failed to convert some data types")

    if train:
        with open("model/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        with open("model/scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        # save df with sorted columns by occurance frequency for each column
        df_sorted = decode_labels_and_scalers(df.copy(), target_single_value=False)
        df_sorted = df_sorted[sorted(df_sorted.columns, key=lambda x: df[x].value_counts().index[0])]
        df_sorted.to_csv("data/sorted_columns.csv", index=False)

        # mix the data
        df = shuffle(df, random_state=33)
        # reset index
        df.reset_index(drop=True, inplace=True)

    # sort columns just like in the 'columns' list
    df = df[columns + [col for col in df.columns if col not in columns]]

    return df


def decode_labels_and_scalers(df, target_single_value=True):
    with open("model/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    if target_single_value:
        df = df.reshape(1, -1)
        return scalers["salary_in_usd"].inverse_transform(df)[0]
    for column in df.columns:
        if df[column].dtype == "int32":
            le = label_encoders[column]
            # Keep as integer until after inverse_transform
            df[column] = le.inverse_transform(df[column].astype(int))
        elif df[column].dtype == "float64":
            scaler = scalers[column]
            transformed = scaler.inverse_transform(df[column].values.reshape(-1, 1))
            df[column] = transformed.ravel().astype("int64")
    return df
