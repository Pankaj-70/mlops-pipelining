import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""Feature Engineering Script:
Loads preprocessed text data, applies TF-IDF vectorization,
and saves train/test feature datasets.
"""

# ----------------------- Logging Setup -----------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------- Load Data -----------------------
def load_data(file_url: str) -> pd.DataFrame:
  """Load CSV data and fill missing values."""
  try:
    df = pd.read_csv(file_url)
    df.fillna('', inplace=True)
    logger.debug("Data loaded and missing values handled")
    return df

  except pd.errors.ParserError as err:
    logger.error(f"Failed to parse file at {file_url}\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during data loading\nError: {err}")
    raise


# ----------------------- TF-IDF Vectorization -----------------------
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
  """Vectorize text columns using TF-IDF and return train/test DataFrames."""
  try:
    vectorizer = TfidfVectorizer(max_features=max_features)

    x_train = train_data['text'].values
    y_train = train_data['target'].values
    x_test = test_data['text'].values
    y_test = test_data['target'].values

    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    train_df = pd.DataFrame(x_train_bow.toarray())
    test_df = pd.DataFrame(x_test_bow.toarray())

    train_df['label'] = y_train
    test_df['label'] = y_test

    logger.debug("Train and test data vectorized")
    return train_df, test_df

  except Exception as err:
    logger.error(f"Unexpected error during TF-IDF vectorization\nError: {err}")
    raise


# ----------------------- Save Data -----------------------
def save_data(df: pd.DataFrame, filePath: str) -> None:
  """Save DataFrame to CSV."""
  try:
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    df.to_csv(filePath, index=False)
    logger.debug(f"File saved at {filePath}")

  except Exception as err:
    logger.error(f"Unexpected error while saving file\nError: {err}")
    raise


# ----------------------- Main Pipeline -----------------------
def main() -> None:
  """Load, vectorize, and save train/test TF-IDF datasets."""
  try:
    max_features = 50
    train_data = load_data("./data/interim/train_preprocessed.csv")
    test_data = load_data("./data/interim/test_preprocessed.csv")

    train_df, test_df = apply_tfidf(train_data, test_data, max_features)

    save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
    save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

  except Exception as err:
    logger.error(f"Unexpected error during feature engineering\nError: {err}")
    raise


# ----------------------- Script Entry -----------------------
if __name__ == "__main__":
  main()
