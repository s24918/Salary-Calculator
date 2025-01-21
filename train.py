import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from src.preprocess import preprocess_data, decode_labels_and_scalers


data_init = pd.read_csv('data/ds_salaries.csv')
print(data_init.info())
data = preprocess_data(data_init, train=True)
print(data.info())

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary_in_usd', axis=1), data['salary_in_usd'], test_size=0.2, random_state=42)

# All categorical features are now encoded as integers
categorical = ['experience_level', 'job_title', 'employee_residence', 'company_location', 'company_size']

# Create dataset with encoded categorical features
train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns), categorical_feature=categorical)

# LightGBMXT: {'learning_rate': 0.05, 'extra_trees': True}
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'root_mean_squared_error',
    'extra_trees': True
}

model = lgb.train(params, train_data, feature_name=list(X_train.columns), categorical_feature=categorical)

# evaluate the model
y_pred = model.predict(X_test)

print(f"Predicted: {y_pred[:5]}")
print(f"Labeled: {y_test[:5]}")

print("RMSE:", root_mean_squared_error(y_test, y_pred)) # 0.1
print("MAE:", mean_absolute_error(y_test, y_pred)) # 0.07

decoded_y_test = pd.DataFrame(y_test, columns=['salary_in_usd'])
decoded_y_pred = pd.DataFrame(y_pred, columns=['salary_in_usd'])
decoded_y_test = decode_labels_and_scalers(decoded_y_test, False)
decoded_y_pred = decode_labels_and_scalers(decoded_y_pred, False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(decoded_y_test, decoded_y_pred, alpha=0.5)
plt.plot([decoded_y_test.min(), decoded_y_test.max()], [decoded_y_test.min(), decoded_y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.tight_layout()
plt.savefig('visualizations/actual_vs_predicted.png')

model.save_model('model/model.pkl')

