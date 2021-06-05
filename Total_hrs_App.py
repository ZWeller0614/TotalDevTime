import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.write(""" 
# Total Recommended Hours
Predict the total hours from Pick up through delivery
""")

df = pd.read_csv("app.csv")
x = df.drop('Total_Hrs', axis=1)
y = df['Total_Hrs']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


def get_user_input():
    Month = st.sidebar.selectbox('Month', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    PickUp_Region = st.sidebar.selectbox('PickUp Region', (1, 2, 3, 4, 5, 6, 7))
    Delivery_Region = st.sidebar.selectbox('Delivery Region', (1, 2, 3, 4, 5, 6, 7))
    Tractor_Weight = st.sidebar.number_input("Weight")
    Miles = st.sidebar.number_input("Miles")

    user_data = {'Month': Month,
                 'PickUp Region': PickUp_Region,
                 'Delivery Region': Delivery_Region,
                 'Weight': Tractor_Weight,
                 'Miles': Miles}

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()
st.subheader('Load Features')
st.write(user_input)

model = LinearRegression(fit_intercept=True)
model.fit(x_train, y_train)

prediction = model.predict(user_input)

st.subheader('Recommended Total Hours:')
st.write(prediction)
