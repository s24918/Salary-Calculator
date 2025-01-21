import streamlit as st
import pandas as pd
from src.preprocess import preprocess_data, decode_labels_and_scalers
import lightgbm as lgb
import pickle
from datetime import datetime
import pycountry

model = lgb.Booster(model_file='model/model.pkl')

def country_code_to_name(code):
    try:
        return pycountry.countries.get(alpha_2=code).name
    except AttributeError:
        return None

def country_name_to_code(name):
    try:
        return pycountry.countries.get(name=name).alpha_2
    except AttributeError:
        return None


def get_column_values(column_name):
    try:
        with open("data/sorted_columns.csv", "r") as f:
            df = pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError("Model files not found. Please train the model first. (run train.py)")

    values = df[column_name].unique()
    # move 'other' to the end
    values = sorted(values, key=lambda x: x == "other")
    return values


def main():
    hide_decoration_bar_style = '''
    <style>
        [data-testid="stDecoration"] {background: #FFFFFF;}
    </style>
    '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
    
    st.sidebar.title("Prediction App")
    selected_option = st.sidebar.radio("Wybierz opcję", [ "Aplikacja", "Autorzy"])

    if selected_option == "Autorzy":
        st.title("Autorzy")
        st.write("Andrii Nahornyi - s24918")
        st.write("Pavlo Khrapko - s25140")
    if selected_option == "Aplikacja":
        st.title("Prediction App")
        st.write("This is a simple prediction app that predicts the salary of a data scientist based on the input features.")
        st.subheader("Enter your inputs below:")

        experience_levels = get_column_values("experience_level") #['SE', 'MI', 'EN', 'EX']
        experience_levels_map_decode = {
            "EN": "Entry-level/Junior",
            "MI": "Mid-level/Intermediate",
            "SE": "Senior-level",
            "EX": "Executive-level/Director"
        }
        experience_levels_map_encode = { v: k for k, v in experience_levels_map_decode.items() }
        
        employee_residences = get_column_values("employee_residence")
        job_titles = get_column_values("job_title")

        company_locations = get_column_values("company_location")
        
        # Create input fields
        job_title = st.selectbox("Job Title", job_titles)
        col1, col2, col3 = st.columns(3)
        with col1:
            experience_level = st.selectbox("Experience Level", [experience_levels_map_decode[level] for level in experience_levels])
        with col2:
            employee_residence = st.selectbox("Employee Residence", [f"{code} ({country_code_to_name(code)})" for code in employee_residences])
            remote_ratio = st.select_slider("Remote Ratio", options=[0, 50, 100])
        with col3:
            company_location = st.selectbox("Company Location", [f"{code} ({country_code_to_name(code)})" for code in company_locations])
            company_size = st.selectbox("Company Size", options=['Small', 'Medium', 'Large'])

        if st.button("Calculate", use_container_width=True):
            with st.spinner("Predicting..."):
                # convert the input to a dataframe and preprocess it
                input_data = {
                    "job_title": [job_title],
                    "experience_level": [experience_levels_map_encode[experience_level]],
                    "employee_residence": [employee_residence.split(" ")[0]],
                    "remote_ratio": [remote_ratio],
                    "company_location": [company_location.split(" ")[0]],
                    "company_size": [company_size[0].upper()]
                }
                input_df = pd.DataFrame(input_data, index=[0])
                input_df['work_year'] = datetime.now().year
                input_df = preprocess_data(input_df)
                #predict the salary
                prediction = model.predict(input_df)[0]
                prediction = decode_labels_and_scalers(prediction)
                st.success(f"The predicted salary is: {int(prediction[0])} USD")


if __name__ == "__main__":
    main()
