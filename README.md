# Emotion Detection

Emotion Detection is a kind of text analysis, which will predict the emotion of the given text. Emotions are labelled as below. This Application detections any one of the following emotions.

- Anger
- Fear
- Joy
- Sadness

I have trained the model from the data set taken from [here](https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html). It has three levels.

1. Data Pre Processing
2. Data Training
3. Data Prediction

## Data Pre Processing

Data pre processing is to make data clean and tokenizing for natural langauge processing(NLP). For this process, I used `spacy` python package. Below are the steps to clean data and stored in csv(text, sentiment, prep_processed_text) format for training.

- Removing stop words
- Lemmatization
- Removing Accented Chars like é, è, â, î or ô, ñ
- Removing hash
- Removing special characters

Run below command to execute `prepare.py` script
`python3 prepare.py`

## Data Training

I have trained the data for four machine learning models.

1. Support Vector Machine (SVM)
2. Naive Bayes(NB)
3. Random Forest(RF)
4. Logistic Regression (LR)

Run below command to execute `train.py` script. Model names are `svm`, `nb`, `rf` and `lr`
`python3 train.py -m model_name`

CSV data file containts three colummns

1. text - Original text
2. sentiment - Sentiment of the text(joy, sadness, angry and fear)
3. prep_processed_text - preprocessed text after applying data preprocessing steps

## Data Prediction

Data prediction from the trainined model. It will take text as input and returns emotion of the given text.
Run below command to execute `predict.py` script. Default model name is LR if not specified
`python3 predict.py [-m model_name]`

## Environment setup

### 1. Virtual environment setup

I prefer to use virtual environment, it will be very easy to manage dependency packages. Run below commands to enable virtula environemnt.

`python3 -m venv .venv`
`source .venv/bin/activate`

### 2. Dependency packages installation

Run below command to install all dependencies.

`pip3 install -r requirements.txt`

### 3. Installing language tokenizer

Run below command to download english language tokenizer
`python -m spacy download en_core_web_sm`

## How to start the server
