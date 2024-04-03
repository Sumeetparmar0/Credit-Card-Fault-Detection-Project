from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_obj

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
import os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data transformation initiated")
            
            numerical_cols = ['LIMIT_BAL', 'SEX','EDUCATION','MARRIAGE','AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1',
            'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']

            categorical_cols = []
            

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())]
                    )
            
            #Column Transformer
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline, numerical_cols)])
            
            logging.info('Datatransformation completed')

            return preprocessor

        except Exception as e:
            logging.info('Exception occured in data transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path, test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Reading the train and test data completed')
            logging.info(f'Training Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_obj()

            target_column="default payment next month"
            drop_coulmn=[target_column]

            #train data
            input_feature_train_df=train_df.drop(drop_coulmn,axis=1)
            target_feature_train_df=train_df[target_column]

            #test data
            input_feature_test_df=test_df.drop(drop_coulmn,axis=1)
            target_feature_test_df=test_df[target_column]

            #Data transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info("Applying preprocessing object on training and testing datasets.")



        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)