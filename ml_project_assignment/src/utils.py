import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        # logging.error(f"Error saving object to file: {file_path}. Error: {str(e)}")
        raise CustomException(e, sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)


            # conf_mat = confusion_matrix(y_test, y_test_pred)
            acc_score = accuracy_score(y_test, y_test_pred)
            # class_report = classification_report(y_test, y_test_pred)
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = acc_score

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)