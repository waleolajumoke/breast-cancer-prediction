# deploy our model using pickle on streamlit 
import pickle
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
data = load_breast_cancer()
sc = StandardScaler()

st.title("Breast Cancer Prediction")
st.header("""
This app predicts whether the breast cancer is benign or malignant.
""")
st.write("Fill the form below to get prediction")

# load data from pickle file and use it for prediction
model = pickle.load(open('model.pkl', 'rb'))
features = data.feature_names

# collect inputs dynamically 
values = []
for i,j in enumerate(features):
    values.append(st.number_input(f"Enter {j}", min_value=0.0))

# display the forms in values 
if st.button("Predict"):
    X = np.array(values).reshape(1,-1)
    X = sc.fit_transform(X)
    prediction = model.predict(X)[0]
    if prediction == 0:
        st.success("The Breast Cancer is Malignant")
    else:
        st.error("The Breast Cancer is Benign")
    
