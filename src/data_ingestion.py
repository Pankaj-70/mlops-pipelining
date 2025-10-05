import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

"""Data Ingestion Script:
This module handles data loading, preprocessing, splitting into train/test sets,
and saving with logging at each stage.
"""

# ----------------------- Logging Setup -----------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

# Log format: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------- Load Data -----------------------
def load_data(data_url: str):
  """Load dataset from the specified CSV URL."""
  try:
    df = pd.read_csv(data_url)
    logger.debug(f"Data loaded from {data_url} successfully")
    return df

  except pd.errors.ParserError as err:
    logger.error(f"Failed to parse the file at {data_url}\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during loading of data at {data_url}\nError: {err}")
    raise


# ----------------------- Preprocess Data -----------------------
def preprocess_data(df: pd.DataFrame):
  """Clean and rename dataset columns."""
  try:
    # Remove unnecessary unnamed columns
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    # Rename key columns for clarity
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    logger.debug("Data preprocessing completed")
    return df

  except KeyError as err:
    logger.error(f"Missing columns, preprocessing error\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during preprocessing\nError: {err}")
    raise


# ----------------------- Save Data -----------------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
  """Save training and testing datasets as CSV files."""
  try:
    raw_data_path = os.path.join(data_path, 'raw')
    os.makedirs(raw_data_path, exist_ok=True)
    # Save train and test data to respective files
    train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
    logger.debug(f"Training and testing data saved successfully at {raw_data_path}")

  except Exception as err:
    logger.error(f"Unexpected error during saving data\nError: {err}")
    raise


# ----------------------- Main Pipeline -----------------------
def main():
  """Main function to run the data ingestion pipeline."""
  try:
    test_size = 0.2
    # Source dataset URL
    data_path = 'https://raw.githubusercontent.com/Pankaj-70/mlops-pipelining/refs/heads/main/experiments/spam.csv'

    # Load, preprocess, and split data
    df = load_data(data_url=data_path)
    final_df = preprocess_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)

    # Save processed train and test data
    save_data(train_data, test_data, data_path='./data')

  except Exception as err:
    logger.error(f"Failed to complete data ingestion process\nError: {err}")
    raise


# ----------------------- Script Entry Point -----------------------
if __name__ == '__main__':
  main()
