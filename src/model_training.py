import logging
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

"""Model Training Script:
Loads TF-IDF features, trains a Random Forest model, and saves it using pickle.
"""

# ----------------------- Logging Setup -----------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------- Load Data -----------------------
def load_data(file_url: str) -> pd.DataFrame:
  """Load CSV data."""
  try:
    df = pd.read_csv(file_url)
    logger.debug(f"Data loaded from {file_url}")
    return df

  except pd.errors.ParserError as err:
    logger.error(f"Failed to parse file at {file_url}\nError: {err}")
    raise

  except FileNotFoundError as err:
    logger.error(f"File not found\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during data loading\nError: {err}")
    raise


# ----------------------- Train Model -----------------------
def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
  """Train a Random Forest classifier."""
  try:
    if x_train.shape[0] != y_train.shape[0]:
      raise ValueError("x_train and y_train must have the same number of samples")
    
    cls = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    logger.debug("Beginning model fitting")
    cls.fit(x_train, y_train)
    logger.debug("Model training completed")
    return cls

  except ValueError as err:
    logger.error(f"Value error during model training\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during model training\nError: {err}")
    raise


# ----------------------- Save Model -----------------------
def save_model(model: RandomForestClassifier, file_path: str) -> None:
  """Save trained model to disk using pickle."""
  try:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
      pickle.dump(model, file)
    logger.debug(f"Model saved to {file_path}")

  except FileNotFoundError as err:
    logger.error(f"File not found\nError: {err}")
    raise

  except Exception as err:
    logger.error(f"Unexpected error during model saving\nError: {err}")
    raise


# ----------------------- Main Pipeline -----------------------
def main() -> None:
  """Load features, train model, and save it."""
  try:
    params = {'n_estimators': 25, 'random_state': 2}
    train_data = load_data("./data/processed/train_tfidf.csv")
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    clf = train_model(x_train, y_train, params)
    model_save_path = 'models/model.pkl'
    save_model(clf, model_save_path)

  except Exception as err:
    logger.error(f"Unexpected error during model training\nError: {err}")
    raise


# ----------------------- Script Entry -----------------------
if __name__ == '__main__':
  main()
