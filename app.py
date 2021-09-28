import streamlit as st
from os import path
import joblib
from sklearn.feature_extraction.text import TfidfTransformer

separator = path.sep


data_path = f"data{separator}models"

# Linear Regression model
lr_model = joblib.load(data_path + separator + "lr.pkl")
lr_count_vect = joblib.load(data_path + separator + "lrVector.pkl")

# SVM model
svm_model = joblib.load(data_path + separator + "svm.pkl")
svm_count_vect = joblib.load(data_path + separator + "svmVector.pkl")

# Naive Bayes model
nb_model = joblib.load(data_path + separator + "nb.pkl")
nb_count_vect = joblib.load(data_path + separator + "nbVector.pkl")

# Random Forest model
rf_model = joblib.load(data_path + separator + "rf.pkl")
rf_count_vect = joblib.load(data_path + separator + "rfVector.pkl")


st.title("Emotional Detection Application")
st.write("Welcome to Emotional Detection Application.")


_input = st.text_input("Enter text")
selected_model = st.selectbox(
    "Select Model Name:",
    ("LR", "SVM", "NB", "RF"),
)


if not _input.isdecimal() and len(_input):
    if not selected_model:
        selected_model = "LR"
    if selected_model.lower() == "lr":
        x_counts = lr_count_vect.transform([_input])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = lr_model.predict(x_tfidf)
    elif selected_model.lower() == "svm":
        x_counts = svm_count_vect.transform([_input])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = svm_model.predict(x_tfidf)
    elif selected_model.lower() == "nb":
        x_counts = nb_count_vect.transform([_input])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = nb_model.predict(x_tfidf)
    elif selected_model.lower() == "rf":
        x_counts = rf_count_vect.transform([_input])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = rf_model.predict(x_tfidf)
    else:
        x_counts = lr_count_vect.transform([_input])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = lr_model.predict(x_tfidf)
    st.write(f"'{_input}' emotion is '{predictions[0]}'")
else:
    pass
    # st.write("Invalid input text.")
