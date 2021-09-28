from sklearn import model_selection, metrics, linear_model, svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import time
import pandas
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.naive_bayes import MultinomialNB
import argparse
import joblib
from os import path

separator = path.sep


ap = argparse.ArgumentParser()
ap.add_argument(
    "-m",
    "--model",
    required=True,
    type=str,
    default="nb",
    choices=["svm", "nb", "lr", "rf"],
    help="specify model name. allowed values are svm, nb, lr, rf",
)
args = vars(ap.parse_args())
# try:
script_start_time = time.time()
input_data_path = "data/csv/"
output_data_path = "data/models/"
# data_path = "../data/textcat/aclImdb/train/"
# Assign variables for command line arguments
inputData = f"{input_data_path}emotions.csv"
# inputData = f"{data_path}sample.csv"
modelName = args["model"]

# x_fld = "final_text"
x_fld = "prep_processed_text"
summary_file = f"{output_data_path}{modelName}_{x_fld}_summary.txt"
y_flds = "sentiment"
# Validate ML Model selection
modelNameList = ["nb", "svm", "lr", "rf"]

# Create output names based on ML Model selection
modelFile = f"{output_data_path}{modelName}.pkl"
countVectorizerFileName = f"{output_data_path}{modelName}Vector.pkl"

# Stage 1. Data Preparation

# Load transactions_file
file_reading_start_time = time.time()
print_str = ""
df = pandas.read_csv(inputData)
print_str += "Input file reading completed in {0} seconds.\n".format(
    round(time.time() - file_reading_start_time, 3)
)
df = shuffle(df)
# Split data frame for training and validation
train_x, valid_x, train_y, valid_y = train_test_split(
    df[x_fld].values.astype("U"), df[y_flds], train_size=0.8
)

# Print total rows in all data frames
print_str += f"Trained transactions:{train_x.shape[0]}  rows.\n"
print_str += f"Testing transactions:{valid_y.shape[0]}  rows.\n"
# print("Total transactions:", df[1].shape[0], " rows.")
#

# Stage 2. Feature Engineering
count_vect = CountVectorizer(ngram_range=(1, 10))
X_train_counts = count_vect.fit_transform(train_x)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_valid_counts = count_vect.transform(valid_x)
X_valid_tfidf = tfidf_transformer.transform(X_valid_counts)

# Stage 3. Model Training

# Trigger training, based on ML Model selection
print("Training in progress...")
print_str += "Training in progress...\n"
training_start_time = time.time()
if modelName.lower() == "nb":
    print_str += "ML Model: NB\n"
    clf = MultinomialNB().fit(X_train_tfidf, train_y)
elif modelName.lower() == "svm":
    print_str += "ML Model: SVM\n"
    clf = svm.SVC().fit(X_train_tfidf, train_y)
elif modelName.lower() == "lr":
    print("ML Model: LR")
    print_str += "ML Model: LR\n"
    clf = linear_model.LogisticRegression().fit(X_train_tfidf, train_y)
elif modelName.lower() == "rf":
    print_str += "ML Model: RF\n"
    clf = ensemble.RandomForestClassifier().fit(X_train_tfidf, train_y)
print_str += "Training completed in {0} seconds.\n".format(
    round(time.time() - training_start_time, 3)
)

predict_start_time = time.time()
# Trigger prediction, based on on validation data
predictions = clf.predict(X_valid_tfidf)
print_str += "Prediction in progress...\n"
print("Prediction in progress...")
# print(predictions)

# Compute accuracy score
accuracy = metrics.accuracy_score(predictions, valid_y)
predict_end_time = time.time()
print_str += f"Prediction completed in {round(predict_end_time - predict_start_time, 3)} seconds.\n"
# Print accuracy score
print_str += f"Accuracy Score:{accuracy}\n"
print(f"Accuracy Score:{accuracy}")

# Save model file
joblib.dump(clf, modelFile)
print("Trained Model file:", modelFile)

# Save count vectorizer file
joblib.dump(count_vect, countVectorizerFileName)
print("Trained Count Vectorizer file:", countVectorizerFileName)
# except Exception as e:
#     print("Exception:", e)
# exit(2)
script_end_time = time.time()
print_str += (
    f"Script completed in {round(script_end_time - script_start_time, 3)} seconds.\n"
)
print(print_str)
fs = open(summary_file, "w")
fs.write(print_str)
fs.close()
