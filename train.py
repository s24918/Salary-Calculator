import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocess import preprocess_data


data_init = pd.read_csv('data/ds_salaries.csv')
print(data_init.info())
data = preprocess_data(data_init)
print(data.info())

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary_in_usd', axis=1), data['salary_in_usd'], test_size=0.3, random_state=42)

# categorical = experience_level: object, employment_type: object, job_title: object, salary_currency: object, employee_residence: object, company_location: object, company_size: object
categorical = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'employee_residence', 'company_location', 'company_size']

# Define the training data
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical)

# Set the parameters for the LightGBMXT model
# LightGBMXT: {'learning_rate': 0.05, 'extra_trees': True}
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'root_mean_squared_error',
    'extra_trees': True
}

# Train the LightGBMXT model
model = lgb.train(params, train_data)

# evaluate the model
y_pred = model.predict(X_test)

# compare the predictions with the actual values
print(f"train: {y_pred[:5]}")
print(f"test: {y_test[:5]}")

# print RMSE
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# save the model
model.save_model('model/model.pkl')