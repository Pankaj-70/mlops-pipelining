import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

"""Data Preprocessing Script:
This module performs text cleaning, stemming, label encoding,
and saves preprocessed data for training and testing.
"""

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# ----------------------- Logging Setup -----------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

# Log format: timestamp - logger name - log level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------- Text Transformation -----------------------
def transform_text(text):
  """Clean, tokenize, remove stopwords/punctuation, and stem input text."""
  ps = PorterStemmer()
  text = text.lower()  # convert to lowercase
  text = nltk.word_tokenize(text)  # tokenize text
  text = [word for word in text if word.isalnum()]  # keep alphanumeric words
  text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
  text = [ps.stem(word) for word in text]  # stemming
  return " ".join(text)


# ----------------------- Data Preprocessing -----------------------
def preprocess_text(df, text_column='text', target_column='target'):
  """Encode target labels, remove duplicates, and clean text column."""
  try:
    logger.debug("Preprocessing started in preprocessing.py")

    # Encode labels (spam/ham → 1/0)
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])
    logger.debug("Label encoding completed")

    # Remove duplicate rows
    df = df.drop_duplicates(keep='first')
    logger.debug("Duplicates removed")

    # Apply text transformation to text column
    df.loc[:, text_column] = df[text_column].apply(transform_text)
    logger.debug("Text column transformed successfully")

    return df

  except KeyError as err:
    logger.error(f"Missing columns, preprocessing.py error\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during text normalization\nError: {err}")
    raise


# ----------------------- Main Function -----------------------
def main(text_column='text', target_column='target'):
  """Main function to load, preprocess, and save train/test datasets."""
  try:
    # Load raw training and testing data
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.debug("Data loaded successfully")

    # Preprocess text columns
    train_processed_data = preprocess_text(train_data, text_column, target_column)
    test_processed_data = preprocess_text(test_data, text_column, target_column)

    # Save processed data
    data_path = os.path.join("./data", "interim")
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)

    logger.debug(f"Processed data saved to {data_path}")

  except FileNotFoundError as err:
    logger.error(f"File not found\nError: {err}")
    raise

  except pd.errors.EmptyDataError as err:
    logger.error(f"No data in file\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during data transformation\nError: {err}")
    raise


# ----------------------- Script Entry Point -----------------------
if __name__ == "__main__":
  main()
