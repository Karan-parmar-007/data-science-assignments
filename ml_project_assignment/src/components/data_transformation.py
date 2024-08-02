import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

# Data transformations config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self, input_feature_train_df):
        try:
            logging.info('Data Transformation initiated')

            logging.info('Pipeline Initiated')

            # Define numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Create the preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, input_feature_train_df.columns.tolist())
                ]
            )

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            target_column_name = 'target'
            drop_columns = [target_column_name]

            # Features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get preprocessing object with correct columns
            preprocessing_obj = self.get_data_transformation_object(input_feature_train_df)

            # Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Processor pickle is created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
