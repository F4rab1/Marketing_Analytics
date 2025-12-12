# %%
from pathlib import Path
import dill
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# parent directory to path to import src module
sys.path.insert(0, str(Path.cwd().parent))

import src.unimib_snowit_project.utils as u

# %%
# Base Params

DATA_IN_DIR = 'data_input'
REVIEWS_IN_FILENAME = 'reviews.csv'                
REVIEWS_LABELLED_IN_FILENAME = 'reviews_labelled.csv' 

DATA_PKL_DIR = 'data_loaded'

REVIEWS_PKL_FILENAME = 'reviews.pkl'                 
REVIEWS_LABELLED_PKL_FILENAME = 'reviews_labelled.pkl'  

NA_VALUES = ['', ' ', '""',
             '#N/A', '#N/A N/A', '#NA', 'N/A', '<NA>', 'n/a', # 'NA',
             '-1.#IND', '1.#IND',
             '-1.#QNAN', '-NaN', '-nan', '-NAN', '1.#QNAN', 'NaN', 'nan', 'NAN',
             'NULL', 'Null', 'null',
             'NONE', 'None', 'none',
             ]

# %%
# Base paths

root_dir_path = u.get_root_dir()

data_in_dir_path = root_dir_path.joinpath(DATA_IN_DIR)
reviews_in_path = data_in_dir_path.joinpath(REVIEWS_IN_FILENAME)               
reviews_labelled_in_path = data_in_dir_path.joinpath(REVIEWS_LABELLED_IN_FILENAME)

data_pkl_dir_path = root_dir_path.joinpath(DATA_PKL_DIR)
reviews_pkl_path = data_pkl_dir_path.joinpath(REVIEWS_PKL_FILENAME)                
reviews_labelled_pkl_path = data_pkl_dir_path.joinpath(REVIEWS_LABELLED_PKL_FILENAME) 

# %% [markdown]
# ## Load Reviews

# %%
safeload_reviews_df = pd.read_csv(reviews_in_path,
                                  dtype="string",
                                  na_values=[],
                                  keep_default_na=False
                                  )

# %%
safeload_reviews_df.columns

# %%
reviews_df = pd.read_csv(reviews_in_path,
                         keep_default_na=False,
                         na_values=NA_VALUES,
                         dtype={
                             "review.uid": "string",   
                             "user.uid": "string",     
                             "text": "string"          
                         }
                         )

reviews_df["text"] = (
    reviews_df["text"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# %%
# CHECK PK VALIDITY

# SELECT count(1) as num_rows
# FROM reviews_df
# WHERE review.uid IS NULL

print(
    reviews_df
    .loc[lambda tbl: tbl["review.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT review.uid, count(1) as num_rows
# FROM reviews_df
# GROUP BY review.uid
# HAVING num_rows > 1

print(
    reviews_df
    .assign(aux=1.0)
    .groupby(["review.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %% [markdown]
# ## Load Reviews Labelled

# %%
safeload_reviews_labelled_df = pd.read_csv(reviews_labelled_in_path,
                                           dtype="string",
                                           na_values=[],
                                           keep_default_na=False
                                           )

safeload_reviews_labelled_df.columns

# %%
reviews_labelled_df = pd.read_csv(
    reviews_labelled_in_path,
    keep_default_na=False,
    na_values=NA_VALUES,
    dtype={
        "labelled_review.uid": "string",   
        "text": "string",                  
        "sentiment_label": "string"        
    }
)

reviews_labelled_df["text"] = (
    reviews_labelled_df["text"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

reviews_labelled_df["sentiment_label"] = (
    reviews_labelled_df["sentiment_label"]
    .str.strip()
    .str.lower()
)

# %%
# CHECK PK VALIDITY

# SELECT count(1)
# FROM reviews_labelled_df
# WHERE labelled_review.uid IS NULL

print(
    reviews_labelled_df
    .loc[lambda tbl: tbl["labelled_review.uid"].isnull()]
    .assign(aux=1.0)
    .shape[0]
)

# SELECT labelled_review.uid, count(1)
# FROM reviews_labelled_df
# GROUP BY labelled_review.uid
# HAVING count(1) > 1

print(
    reviews_labelled_df
    .assign(aux=1.0)
    .groupby(["labelled_review.uid"], dropna=False)
    .agg(num_rows=("aux", pd.Series.count))
    .loc[lambda tbl: tbl["num_rows"] > 1]
)

# %%
with reviews_pkl_path.open('wb') as fh:
    dill.dump(reviews_df, fh)
print(f"Save reviews data in {reviews_pkl_path.as_posix()}")

with reviews_labelled_pkl_path.open('wb') as fh:
    dill.dump(reviews_labelled_df, fh)
print(f"Save reviews labelled data in {reviews_labelled_pkl_path.as_posix()}")

# %%
DATA_PKL_DIR = "data_loaded"

# Filenames
REVIEWS_PKL_FILENAME = "reviews.pkl"              
REVIEWS_LABELLED_PKL_FILENAME = "reviews_labelled.pkl" 

# Root directory
root_dir_path = u.get_root_dir()

# Base PKL directory
data_pkl_dir_path = root_dir_path / DATA_PKL_DIR

# PKL paths
reviews_pkl_path = data_pkl_dir_path / REVIEWS_PKL_FILENAME             
reviews_labelled_pkl_path = data_pkl_dir_path / REVIEWS_LABELLED_PKL_FILENAME 

# Loader
def load_pkl(pkl_path):
    with pkl_path.open("rb") as fh:
        return dill.load(fh)

# Load DataFrames
reviews_df = load_pkl(reviews_pkl_path)                       
reviews_labelled_df = load_pkl(reviews_labelled_pkl_path)     

# %%
reviews_df

# %%
reviews_labelled_df

# %% [markdown]
# ## Cleaning

# %%
import re

def clean_review_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""

    # 1) lowercase
    text = text.lower()

    # 2) remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3) remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 4) remove non-alphabetic characters (keep letters, numbers, punctuation)
    text = re.sub(r"[^a-z0-9.,!?;:'\"()\s]", " ", text)

    # 5) normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6) normalize repeated punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    return text


reviews_df["text"] = reviews_df["text"].apply(clean_review_text)
reviews_labelled_df["text"] = reviews_labelled_df["text"].apply(clean_review_text)

reviews_labelled_df = reviews_labelled_df.loc[
    reviews_labelled_df["text"].str.len() > 3
].reset_index(drop=True)

# %%
sentiment_df = reviews_labelled_df[
    ["labelled_review.uid", "text", "sentiment_label"]
].copy()

sentiment_df = sentiment_df.rename(columns={"labelled_review.uid": "id"})

# %%
X = sentiment_df["text"]
y = sentiment_df["sentiment_label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Train matrix shape:", X_train_tfidf.shape)
print("Test matrix shape:", X_test_tfidf.shape)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

final_clf = LogisticRegression(
    max_iter=1500,
    class_weight="balanced"
)

final_clf.fit(X_train_tfidf, y_train)

y_pred = final_clf.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))

# %%
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=final_clf.classes_,
            yticklabels=final_clf.classes_,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Logistic Regression")
plt.show()

# %%
reviews_df

# %% [markdown]
# ## Sentiment predictions for all reviews using the trained model.

# %%
reviews_tfidf = tfidf.transform(reviews_df["text"])

reviews_df["sentiment_pred"] = final_clf.predict(reviews_tfidf)

# %%
reviews_df.head()


