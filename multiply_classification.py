# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 10:13:47 2026

@author: Lab
"""
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
loan_model = pickle.load(open(os.path.join(BASE_DIR, 'loan_model.sav'), 'rb'))
ridingmower_model = pickle.load(open(os.path.join(BASE_DIR, 'RidingMowers_model.sav'), 'rb'))
bmi_model = pickle.load(open(os.path.join(BASE_DIR, 'bmi_model.sav'), 'rb'))

with st.sidebar:
    selected = option_menu(
        'Classification',
        ['Loan', 'Riding', 'BMI'])


if selected == 'Riding':
    st.title('RidingMowers Predict')

    Income = st.text_input('Income')
    LotSize = st.text_input('LotSize')
    ridingmowers_predict = ''

    if st.button('ridingmowers_predict'):
        ridingmowers_predict = ridingmower_model.predict([[
            float(Income),
            float(LotSize)
        ]])
        if ridingmowers_predict[0] == 0:
            ridingmowers_predict = 'Not Owner'
        else:
            ridingmowers_predict = 'Owner'
    st.success(ridingmowers_predict)


if selected == 'Loan':
    st.title('Loan Predict')

    person_age = st.text_input('person_age')
    person_gender = st.text_input('person_gender')
    person_education = st.text_input('person_education')
    person_income = st.text_input('person_income')
    person_emp_exp = st.text_input('person_emp_exp')
    person_home_ownership = st.text_input('person_home_ownership')
    loan_amnt = st.text_input('loan_amnt')
    loan_intent = st.text_input('loan_intent')
    loan_int_rate = st.text_input('loan_int_rate')
    loan_percent_income = st.text_input('loan_percent_income')
    cb_person_cred_hist_length = st.text_input('cb_person_cred_hist_length')
    credit_score = st.text_input('credit_score')
    previous_loan_defaults_on_file = st.text_input('previous_loan_defaults_on_file')

    loan_predict = ''

    if st.button('Predict'):
        loan_predict = loan_model.predict([[
            float(person_age),
            float(person_gender),
            float(person_education),
            float(person_income),
            float(person_emp_exp),
            float(person_home_ownership),
            float(loan_amnt),
            float(loan_intent),
            float(loan_int_rate),
            float(loan_percent_income),
            float(cb_person_cred_hist_length),
            float(credit_score),
            float(previous_loan_defaults_on_file)
        ]])
        if loan_predict[0] == 0:
            loan_predict = 'Not Accept'
        else:
            loan_predict = 'Accept'
    st.success(loan_predict)


if selected == 'BMI':
    st.title('BMI Classification')

    # BMI class mapping: 0=Extremely Weak, 1=Weak, 2=Normal, 3=Overweight, 4=Obesity, 5=Extreme Obesity
    bmi_classes = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }

    gender = st.selectbox('Gender', ['Male', 'Female'])
    height_cm = st.text_input('Height (cm)', value='160')
    weight_kg = st.text_input('Weight (kg)', value='100')

    bmi_result = ''

    if st.button('Predict'):
        try:
            gender_val = 1 if gender == 'Male' else 0
            height = float(height_cm)
            weight = float(weight_kg)

            prediction = bmi_model.predict([[gender_val, height, weight]])
            bmi_category = bmi_classes[prediction[0]]
            bmi_result = f'Predicted BMI Category: {bmi_category}'
        except ValueError:
            bmi_result = 'Please enter valid numbers for Height and Weight'

    if bmi_result:
        st.success(bmi_result)
