# Salary Calculator

This project aims to create a predictive salary calculator based on a dataset of data science salaries, using various factors like experience level, job role, and company size to estimate employee compensation. The tool leverages historical data to provide insights into how different job-related variables influence salary levels, helping organizations and individuals understand potential earnings in the data science field.

## Dataset
The dataset used for this project is the [2023 Data Science Salaries dataset](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023), available on Kaggle. It includes key information about job roles, locations, experience levels, and other factors that impact salary.

## Team
- Andrii Nahornyi - s24918
- Pavlo Khrapko - s25140

## Project Stages

### Stage 1: Data Cleaning and Preprocessing
The first stage involved checking the data quality, handling any missing values, and removing unnecessary columns to optimize the dataset for modeling.

### Model Features
We selected the following attributes to predict salaries effectively:

- **job_title**: Job role of the employee
- **experience_level**: Employee experience level with possible values:
  - EN (Entry-level/Junior)
  - MI (Mid-level/Intermediate)
  - SE (Senior-level)
  - EX (Executive-level/Director)
- **employee_residence**: Primary country of residence (ISO country code)
- **company_location**: Location of the company’s headquarters or branch where the contract was made (ISO country code)
- **remote_ratio**: Percentage of work done remotely
- **company_size**: Average number of employees in the company:
  - S (small) - fewer than 50 employees
  - M (medium) - 50 to 250 employees
  - L (large) - more than 250 employees
- **work_year**: Year in which the salary was paid (automatically set to the current year when predicting)

### Removed Columns
To streamline the dataset, we removed these columns:
- `salary`
- `salary_currency`
- `employment_type`

## Research jupiter: https://colab.research.google.com/drive/1mIyVmtc3XKCYgpXhdIiblxZ0Ww_HxWtR?usp=sharing#scrollTo=QquoUfNhbFbY

## Prototype
### Running instructions

1. Clone the repo
```
git clone https://github.com/s24918/Salary-Calculator.git
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Train the model
```
python train.py
```
4. Run the app
```
streamlit run calculator.py
```
