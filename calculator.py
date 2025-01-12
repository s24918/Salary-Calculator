import streamlit as st
import pandas as pd
from src.preprocess import preprocess_data, decode_labels_and_scalers
import lightgbm as lgb
import pickle
from datetime import datetime

def get_column_values(column_name):
    try:
        with open("model/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open("model/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Model files not found. Please train the model first. (run train.py)")

    try:
        le = label_encoders[column_name]
        values = le.classes_
    except KeyError:
        values = scalers[column_name].data_max_
    return values

def main():
    st.sidebar.title("Panel boczny")
    selected_option = st.sidebar.radio("Wybierz opcję", [ "Aplikacja", "Autorzy"])

    if selected_option == "Autorzy":
        st.title("Autorzy")
        st.write("Andrii Nahornyi - s24918")
        st.write("Pavlo Khrapko - s25140")
    if selected_option == "Aplikacja":
        st.title("Prediction App")
        st.subheader("Enter your inputs below:")

        experience_levels = get_column_values("experience_level") #['SE', 'MI', 'EN', 'EX']
        #employment_types = ['FT', 'CT', 'FL', 'PT']
        employee_residences = get_column_values("employee_residence")
        # ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'PT', 'NL', 'CH', 'CF', 'FR', 'AU',
        #                        'FI', 'UA', 'IE', 'IL', 'GH', 'AT', 'CO', 'SG', 'SE', 'SI', 'MX', 'UZ', 'BR', 'TH',
        #                        'HR', 'PL', 'KW', 'VN', 'CY', 'AR', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK',
        #                        'IT', 'MA', 'LT', 'BE', 'AS', 'IR', 'HU', 'SK', 'CN', 'CZ', 'CR', 'TR', 'CL', 'PR',
        #                        'DK', 'BO', 'PH', 'DO', 'EG', 'ID', 'AE', 'MY', 'JP', 'EE', 'HN', 'TN', 'RU', 'DZ',
        #                        'IQ', 'BG', 'JE', 'RS', 'NZ', 'MD', 'LU', 'MT']
        job_titles = get_column_values("job_title")
     
        # ['Principal Data Scientist', 'ML Engineer', 'Data Scientist',
        #               'Applied Scientist', 'Data Analyst', 'Data Modeler', 'Research Engineer',
        #               'Analytics Engineer', 'Business Intelligence Engineer',
        #               'Machine Learning Engineer', 'Data Strategist', 'Data Engineer',
        #               'Computer Vision Engineer', 'Data Quality Analyst',
        #               'Compliance Data Analyst', 'Data Architect',
        #               'Applied Machine Learning Engineer', 'AI Developer', 'Research Scientist',
        #               'Data Analytics Manager', 'Business Data Analyst', 'Applied Data Scientist',
        #               'Staff Data Analyst', 'ETL Engineer', 'Data DevOps Engineer', 'Head of Data',
        #               'Data Science Manager', 'Data Manager', 'Machine Learning Researcher',
        #               'Big Data Engineer', 'Data Specialist', 'Lead Data Analyst',
        #               'BI Data Engineer', 'Director of Data Science',
        #               'Machine Learning Scientist', 'MLOps Engineer', 'AI Scientist',
        #               'Autonomous Vehicle Technician', 'Applied Machine Learning Scientist',
        #               'Lead Data Scientist', 'Cloud Database Engineer', 'Financial Data Analyst',
        #               'Data Infrastructure Engineer', 'Software Data Engineer', 'AI Programmer',
        #               'Data Operations Engineer', 'BI Developer', 'Data Science Lead',
        #               'Deep Learning Researcher', 'BI Analyst', 'Data Science Consultant',
        #               'Data Analytics Specialist', 'Machine Learning Infrastructure Engineer',
        #               'BI Data Analyst', 'Head of Data Science', 'Insight Analyst',
        #               'Deep Learning Engineer', 'Machine Learning Software Engineer',
        #               'Big Data Architect', 'Product Data Analyst',
        #               'Computer Vision Software Engineer', 'Azure Data Engineer',
        #               'Marketing Data Engineer', 'Data Analytics Lead', 'Data Lead',
        #               'Data Science Engineer', 'Machine Learning Research Engineer',
        #               'NLP Engineer', 'Manager Data Management', 'Machine Learning Developer',
        #               '3D Computer Vision Researcher', 'Principal Machine Learning Engineer',
        #               'Data Analytics Engineer', 'Data Analytics Consultant',
        #               'Data Management Specialist', 'Data Science Tech Lead',
        #               'Data Scientist Lead', 'Cloud Data Engineer', 'Data Operations Analyst',
        #               'Marketing Data Analyst', 'Power BI Developer', 'Product Data Scientist',
        #               'Principal Data Architect', 'Machine Learning Manager',
        #               'Lead Machine Learning Engineer', 'ETL Developer', 'Cloud Data Architect',
        #               'Lead Data Engineer', 'Head of Machine Learning', 'Principal Data Analyst',
        #               'Principal Data Engineer', 'Staff Data Scientist', 'Finance Data Analyst']

        company_locations = get_column_values("company_location")
        # ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'NL', 'CH', 'CF', 'FR', 'FI', 'UA',
        #                      'IE', 'IL', 'GH', 'CO', 'SG', 'AU', 'SE', 'SI', 'MX', 'BR', 'PT', 'RU', 'TH', 'HR',
        #                      'VN', 'EE', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK', 'IT', 'MA', 'PL', 'AL',
        #                      'AR', 'LT', 'AS', 'CR', 'IR', 'BS', 'HU', 'AT', 'SK', 'CZ', 'TR', 'PR', 'DK', 'BO',
        #                      'PH', 'BE', 'ID', 'EG', 'AE', 'LU', 'MY', 'HN', 'JP', 'DZ', 'IQ', 'CN', 'NZ', 'CL',
        #                      'MD', 'MT']
        
        # sort job titles by alphabetical order
        job_titles.sort()
        # Create input fields
        job_title = st.selectbox("Job Title", job_titles)
        col1, col2, col3 = st.columns(3)
        with col1:
            experience_level = st.selectbox("Experience Level", experience_levels)
        with col2:
            employee_residence = st.selectbox("Employee Residence", employee_residences)
            remote_ratio = st.select_slider("Remote Ratio", options=[0, 50, 100])
        with col3:
            company_location = st.selectbox("Company Location", company_locations)
            company_size = st.selectbox("Company Size", options=['L', 'S', 'M'])

        if st.button("Predict", use_container_width=True):
            with st.spinner("Predicting..."):
                # convert the input to a dataframe and preprocess it
                input_data = {
                    "job_title": [job_title],
                    "experience_level": [experience_level],
                    "employee_residence": [employee_residence],
                    "remote_ratio": [remote_ratio],
                    "company_location": [company_location],
                    "company_size": [company_size]
                }
                input_df = pd.DataFrame(input_data)
                input_df['work_year'] = datetime.now().year
                input_df = preprocess_data(input_df)
                #predict the salary
                model = lgb.Booster(model_file='model/model.pkl')
                prediction = model.predict(input_df)[0]
                prediction = decode_labels_and_scalers(prediction)
                st.success(f"The predicted salary is: {int(prediction[0])} USD")


if __name__ == "__main__":
    main()
