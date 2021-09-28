from sklearn.feature_extraction.text import TfidfTransformer
import joblib
import argparse


ap = argparse.ArgumentParser()
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="lr",
    choices=["lr", "svm", "nb", "rf"],
    help="specify model name. allowed values are svm, nb, lr, rf",
)
args = vars(ap.parse_args())
modelName = args["model"]

data_path = "data/models/"
input_str = "Good"

# Loading  model
modelFile = ""
countFile = ""

if modelName.lower() == "nb":
    modelFile = data_path + "nb.pkl"
    countFile = data_path + "nbVector.pkl"
elif modelName.lower() == "svm":
    modelFile = data_path + "svm.pkl"
    countFile = data_path + "svmVector.pkl"
elif modelName.lower() == "lr":
    modelFile = data_path + "lr.pkl"
    countFile = data_path + "lrVector.pkl"
elif modelName.lower() == "rf":
    modelFile = data_path + "rf.pkl"
    countFile = data_path + "rfVector.pkl"

model = joblib.load(modelFile)
count_vect = joblib.load(countFile)

x_counts = count_vect.transform([input_str])
tfidf_transformer = TfidfTransformer()
x_tfidf = tfidf_transformer.fit_transform(x_counts)

# Trigger prediction
predictions = model.predict(x_tfidf)

print(f"'{input_str}' emotion is '{predictions[0]}'")
