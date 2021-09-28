from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from os import path

separator = path.sep

app = Flask(__name__, template_folder=".")


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


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST", "PUT"])
def predict_api():
    input_text = request.args.get("text")
    if not input_text:
        return (
            jsonify({"message": "text parameter is required"}),
            400,
        )

    input_model = request.args.get("model", "LR")
    if input_model.lower() == "lr":
        print("LR model")
        x_counts = lr_count_vect.transform([input_text])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = lr_model.predict(x_tfidf)
    elif input_model.lower() == "svm":
        print("SVM model")
        x_counts = svm_count_vect.transform([input_text])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = svm_model.predict(x_tfidf)
    elif input_model.lower() == "nb":
        print("NB model")
        x_counts = nb_count_vect.transform([input_text])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = nb_model.predict(x_tfidf)
    elif input_model.lower() == "rf":
        print("RF model")
        x_counts = rf_count_vect.transform([input_text])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = rf_model.predict(x_tfidf)
    else:
        print("Default LR model")
        x_counts = lr_count_vect.transform([input_text])
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        predictions = lr_model.predict(x_tfidf)

    return (
        jsonify({"text": input_text, "emotion": predictions[0], "model": input_model}),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898)
