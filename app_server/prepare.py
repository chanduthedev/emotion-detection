import pandas as pd
import spacy
from sklearn.utils import shuffle
import unidecode
from os import path

separator = path.sep


# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
txt_data_path = "data/txt"
csv_data_path = "data/csv"

input_file = f"{txt_data_path}{separator}sadness.txt"

# for splitting and writing to df  as csv
# word_list = []
# sentense_list = []
# sentiment_list = []
# with open(input_file, "r") as reader:
#     line = reader.readline()
#     while line:
#         words = line.split("\t")
#         if len(words) > 2:
#             sentense_list.append(words[1])
#             sentiment_list.append(words[2])
#             # word_list.append({"text": words[1], "sentiment": words[2]})
#             # print(f"{words[0]}:{words[2]}")
#         line = reader.readline()

# data = {"text": sentense_list, "sentiment": sentiment_list}
# df = pd.DataFrame(data)
# df.to_csv(f"{data_path}/sadness.csv", columns=df.columns, index=False)
# print(df)

input_file = f"{csv_data_path}sadness.csv"
sadness_df = pd.read_csv(f"{csv_data_path}{separator}sadness.csv")
joy_df = pd.read_csv(f"{csv_data_path}{separator}joy.csv")
anger_df = pd.read_csv(f"{csv_data_path}{separator}anger.csv")
fear_df = pd.read_csv(f"{csv_data_path}{separator}fear.csv")

emotions_list = [joy_df, anger_df, fear_df, sadness_df]
emotions_df = pd.concat(emotions_list)


def pre_process(text):
    # TODO need to remove special characters like ., ? etc.
    # TextBlob(text).words
    doc = nlp(text)
    # token_list = [token for token in doc]
    no_stop_watch__tokens = [token for token in doc if not token.is_stop]

    lemma_tokens = [token.lemma_ for token in no_stop_watch__tokens]

    # Remove Accented Chars: The most common accents are
    # the acute (é), grave (è), circumflex (â, î or ô), tilde (ñ)
    unaccented_data = [unidecode.unidecode(token) for token in lemma_tokens]

    no_hash_data = [token.replace("#", "") for token in unaccented_data]

    # Remove special charcters
    final_text = " ".join(token for token in no_hash_data if token.isalnum())
    return final_text


print("Data pre-processing is in progress...")
emotions_df["final_text"] = emotions_df["text"].apply(lambda x: pre_process(x))
print("Data pre-processing completed.")
emotions_df = shuffle(emotions_df)
emotions_df.to_csv(
    f"{csv_data_path}{separator}emotions.csv", columns=emotions_df.columns, index=False
)
