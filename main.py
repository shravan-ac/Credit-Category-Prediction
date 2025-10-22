import numpy as np
import pandas as pd
import streamlit as st
import pickle

with open('credit.pkl','rb') as f:
    model_data=pickle.load(f)

model=model_data['model']
encoders=model_data['encoders']
scaler=model_data['scaler']
feature_names=model_data['features']

st.title("Credit Score Status Prediction")

num_bank_accounts=st.number_input("Num_Bank_Accounts",min_value=0,max_value=10)
interest_rate=st.number_input("Interest_Rate",min_value=0,max_value=100)
num_of_loan=st.number_input("Num_of_Loan",min_value=0,max_value=10)
delay_from_due=st.number_input("Delay_from_due_date",min_value=0,max_value=70)
num_of_delayed_payment=st.number_input("Num_of_Delayed_Payment",min_value=0,max_value=28)
changed_credit_limit=st.number_input("Changed_Credit_Limit",min_value=0.00,max_value=36.00)
num_credit_inquiries=st.number_input("Num_Credit_Inquiries",min_value=0.00,max_value=17.00)
outstanding_debt=st.number_input("Outstanding_Debt",min_value=0.00,max_value=4000.00)

st.write("Credit History Age :")
credit_history_years=st.selectbox("Years",options=list(range(0,35)))
credit_history_months=st.selectbox("Months",options=list(range(0,12)))
credit_history_age=credit_history_years*12 + credit_history_months

payment_of_min_amount=st.selectbox("Payment_of_Min_Amount",["Yes","No","NM"])

input_df=pd.DataFrame({"Num_Bank_Accounts":[num_bank_accounts],"Interest_Rate":[interest_rate],"Num_of_Loan":[num_of_loan],"Delay_from_due_date":[delay_from_due],"Num_of_Delayed_Payment":[num_of_delayed_payment],"Changed_Credit_Limit":[changed_credit_limit],"Num_Credit_Inquiries":[num_credit_inquiries],"Outstanding_Debt":[outstanding_debt],"Credit_History_Age":[credit_history_age],"Payment_of_Min_Amount":[payment_of_min_amount]})

input_df['Payment_of_Min_Amount']=encoders['Payment_of_min'].transform(input_df['Payment_of_Min_Amount'])
input_df=input_df[feature_names]

if st.button("Predict"):
    y_pred=model.predict(input_df)
    pred=encoders['Credit_mix'].inverse_transform([y_pred])[0]
    st.success(f"Prediction={pred[0]}")
    st.balloons()