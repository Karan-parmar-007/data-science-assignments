import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pymongo import MongoClient

# Initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact', 'train.csv')
    test_data_path = os.path.join('artifact', 'test.csv')
    raw_data_path = os.path.join('artifact', 'raw.csv')

# Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion')

        try:
            client = MongoClient('url')
            db = client['ml_project_preprocessed']  # Create or connect to a database
            collection = db['sklearn_breast_cancer_preprocessed']  # Create or connect to a collection
            data = collection.find() 
            data_list = list(data)
            df = pd.DataFrame(data_list)
            df = df.drop(labels=['_id'], axis=1)
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")
            logging.info("Train test split")

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)
